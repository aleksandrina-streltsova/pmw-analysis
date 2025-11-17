import logging
import pathlib
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial import KDTree
from tqdm import tqdm

from pmw_analysis.constants import COLUMN_SUFFIX_QUANT, COLUMN_COUNT, ArgTransform, ArgSurfaceType, DIR_PMW_ANALYSIS, \
    TC_COLUMNS, FILE_DF_FINAL, FILE_DF_FINAL_NEWEST
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.logging import timing


def main():
    logging.basicConfig(level=logging.INFO)

    arg_transform = ArgTransform.V4
    arg_surface_type = ArgSurfaceType.LAND
    df_dir_path = pathlib.Path(DIR_PMW_ANALYSIS) / arg_transform.value / arg_surface_type.value

    transform = get_transformation_function(arg_transform)
    quant_columns = transform(TC_COLUMNS)
    quant_columns_suffixed = [f"{col}{COLUMN_SUFFIX_QUANT}" for col in quant_columns]

    df_final = pl.read_parquet(df_dir_path / FILE_DF_FINAL)
    df_newest = pl.read_parquet(df_dir_path / FILE_DF_FINAL_NEWEST)

    df_newest_agg = df_newest.group_by(quant_columns_suffixed).agg(pl.len().alias(COLUMN_COUNT))[:10000]
    df_newest_agg_nbr_count = count_neighbors(df_final, df_newest_agg, quant_columns, quant_columns_suffixed)

    df_newest_nbr_count = df_newest.join(df_newest_agg_nbr_count, on=quant_columns_suffixed, how="left")
    df_newest_nbr_count.write_parquet(df_dir_path / "final_newest_nbr_count.parquet")


def count_neighbors(df: pl.DataFrame, df_query: pl.DataFrame,
                    quant_columns: list[str], quant_columns_suffixed: list[str]) -> pl.DataFrame:
    df_preprocessed, df_query_preprocessed = _preprocess(df, df_query, quant_columns, quant_columns_suffixed)

    array, array_query = _get_scaled_arrays(df_preprocessed, df_query_preprocessed, quant_columns,
                                            quant_columns_suffixed)

    weight = df_preprocessed[COLUMN_COUNT].to_numpy()

    tree = KDTree(array)

    nbr_count_2diag = _count_neighbors_radius(tree, weight, array_query,
                                              radius=2 * array.shape[1] ** 0.5,
                                              series_name_suffix="_2diag")

    nbr_count_1 = _count_neighbors_radius(tree, weight, array_query,
                                          radius=1,
                                          series_name_suffix="_1")

    df_nbr_count = df_query_preprocessed[quant_columns_suffixed].with_columns(nbr_count_2diag + nbr_count_1)

    return df_query.join(df_nbr_count, on=quant_columns_suffixed, how="left")


def _preprocess(df: pl.DataFrame, df_query: pl.DataFrame,
                quant_columns: list[str], quant_columns_suffixed: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    logging.info(f"Number of rows before removing NaNs: {len(df)}, {len(df_query)}")
    df = df.drop_nans(subset=quant_columns)
    df_query = df_query.drop_nans(subset=quant_columns_suffixed)
    logging.info(f"Number of rows after removing NaNs: {len(df)}, {len(df_query)}")

    return df, df_query


def _get_scaled_arrays(df: pl.DataFrame, df_query: pl.DataFrame,
                       quant_columns: list[str], quant_columns_suffixed: list[str]) -> tuple[np.ndarray, np.ndarray]:
    array = df[quant_columns].to_numpy()
    array_query = df_query[quant_columns_suffixed].to_numpy()

    space_size = 1
    for idx, column in enumerate(quant_columns):
        unique = np.unique(array[:, idx])
        unique = unique[~np.isnan(unique)]
        unique.sort()

        n = len(unique)

        scale = np.median(np.diff(unique))
        space_size *= n

        array[:, idx] = np.round(array[:, idx] / scale)
        array_query[:, idx] = np.round(array_query[:, idx] / scale)

        logging.info(f"{column}: {n}")

    logging.info(f"Space size: {space_size}")

    return array, array_query


def _count_neighbors_radius(tree: KDTree, weight: np.ndarray,
                            array_query: np.ndarray,
                            radius: float,
                            series_name_suffix: str) -> tuple[pl.Series, pl.Series]:
    with timing("Querying points"):
        indices = tree.query_ball_point(array_query, r=radius, workers=16)

        nbr_count, nbr_count_weighted = _count_neighbors_radius_batched(indices, weight)

        for count, title_suffix in [(nbr_count, ""), (nbr_count_weighted, " weighted")]:
            log_bins = np.logspace(0, max(2, np.log(np.nanmax(count) * 1.1)) / np.log(2), base=2)
            log_bins = np.insert(log_bins, 0, 0.0)
            plt.hist(count, bins=log_bins)
            plt.title(f"Neighbor count{title_suffix} {series_name_suffix.strip('_')}")
            plt.xscale("log")
            plt.show()

        return (
            pl.Series(name=f"nbr_count{series_name_suffix}", values=nbr_count),
            pl.Series(name=f"nbr_count_weighted{series_name_suffix}", values=nbr_count_weighted),
        )


def _count_neighbors_radius_batched(indices: Iterable[np.ndarray], weight: np.ndarray, batch_size: int = 10000):
    N = len(indices)

    nbr_count = np.zeros(N, dtype=np.int64)
    nbr_count_weighted = np.zeros(N, dtype=np.int64)

    for start in tqdm(range(0, N, batch_size)):
        end = min(start + batch_size, N)

        # --- flatten this batch ---
        batch_indices = indices[start:end]
        flat_idx = np.concatenate(batch_indices)

        # --- build offsets for this batch ---
        sizes = np.array([len(i) for i in batch_indices])
        offsets = np.cumsum(sizes)

        # --- cumulative sum over count_final for flattened idx ---
        cf = weight[flat_idx]
        cs = np.concatenate(([0], np.cumsum(cf)))

        # --- convert cumulative sum back to per-item sums ---
        starts_arr = np.concatenate(([0], offsets[:-1]))
        nbr_count_weighted[start:end] = cs[offsets] - cs[starts_arr]
        nbr_count[start:end] = sizes

    assert np.all(nbr_count >= 0)
    assert np.all(nbr_count_weighted >= 0)

    return nbr_count, nbr_count_weighted


if __name__ == '__main__':
    main()
