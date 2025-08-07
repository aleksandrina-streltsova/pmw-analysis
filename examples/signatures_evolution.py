"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import pathlib

import faiss
import numpy as np
import polars as pl
from tqdm import tqdm

from pmw_analysis.analysis.spatial_visualization import plot_variables_on_map
from pmw_analysis.constants import COLUMN_LON, COLUMN_LAT, FILE_DF_FINAL_K, \
    DIR_NO_SUN_GLINT, DIR_IMAGES
from pmw_analysis.constants import DIR_PMW_ANALYSIS, TC_COLUMNS, ArgTransform
from pmw_analysis.processing.filter import filter_by_signature_occurrences_count
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict
from pmw_analysis.quantization.script import get_transformation_function

K = 100000


def _calculate_nn_distances(df_all: pl.DataFrame, df_k: pl.DataFrame):
    transform = get_transformation_function(ArgTransform.DEFAULT)
    quant_columns = transform(TC_COLUMNS)
    unc_dict = {col: 10 * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}

    def make_discrete(df: pl.DataFrame):
        df_result = df.select(quant_columns)
        df_result = df_result.with_columns([(pl.col(quant_col) / unc_dict[quant_col]).round().cast(pl.UInt16)
                                            for quant_col in quant_columns])
        return df_result

    array_all = make_discrete(df_all).to_numpy()
    array_k = make_discrete(df_k).to_numpy()
    distances = np.zeros(len(array_k))

    index = faiss.IndexFlatL2(array_all.shape[1])
    index.add(array_all)

    # divide query into m parts to estimate total time more easily
    m = 100
    size = (len(array_k) + m - 1) // m
    for i in tqdm(range(m)):
        l = i * size
        r = min((i + 1) * size, len(array_k))
        distances[l: r] = np.sqrt(index.search(r - l, array_k[l: r], k=2)[0][:, 1])

    return distances


def main():
    arg_transform = ArgTransform.V4
    df_dir_path = pathlib.Path(DIR_PMW_ANALYSIS) / arg_transform.value / DIR_NO_SUN_GLINT
    transform = get_transformation_function(arg_transform)
    quant_columns = transform(TC_COLUMNS)

    images_dir = pathlib.Path(DIR_IMAGES) / "map" / df_dir_path.name
    images_dir.mkdir(parents=True, exist_ok=True)

    # lfs = []
    # dfs = []
    # for i in range(6):
    #     lf_path = pathlib.Path(f"/ltenas8/data/PMW_Analysis/tmp/default/level0/level1/level2/level3/{i}.parquet")
    #     lf = pl.scan_parquet(lf_path)
    #     lfs.append(lf)
    #     lf = lf.drop(AGG_OFF_COLUMNS)
    #     df = lf.collect()
    #     dfs.append(df)
    #
    # df_quantized = merge_quantized_pmw_features(dfs, quant_columns)
    # df_quantized_k = take_k_sorted(df_quantized, COLUMN_OCCURRENCE, K, COLUMN_COUNT, descending=True)
    #
    # # 1. Analyze quantized features' ranges
    # df_quantized_min = df_quantized.select(pl.min(quant_columns))
    # df_quantized_max = df_quantized.select(pl.max(quant_columns))
    #
    # n_rows_with_extrema = df_quantized_k.filter(pl.any_horizontal([
    #     (pl.col(quant_col) == df_quantized_min[quant_col].item())
    #     .or_(pl.col(quant_col) == df_quantized_max[quant_col].item())
    #     for quant_col in quant_columns
    # ]))[COLUMN_COUNT].sum()
    #
    # print(f"{n_rows_with_extrema} / {df_quantized_k[COLUMN_COUNT].sum()} "
    #       f"observations contain extrema of some quantized features")
    #
    # # 2. Check the distance to the closest neighbor
    # distances = _calculate_nn_distances(df_quantized, df_quantized_k)
    # np.save(pathlib.Path(DIR_PMW_ANALYSIS) / "unique_k_nn_distances.npy", distances)
    # distances = np.load(pathlib.Path(DIR_PMW_ANALYSIS) / "unique_k_nn_distances.npy")
    #
    # print(f"{(distances == 1).sum()} / {len(distances)} = {100 * (distances == 1).sum() / len(distances):.2f}% "
    #       f"observations have their closest neighbor at distance 1")
    # plt.hist(distances ** 2, bins=16)
    # plt.yscale("log")
    # plt.grid()
    # if SAVEFIG_FLAG:
    #     plt.savefig(images_dir / "hist_nn_distances.png")
    # plt.show()
    #
    # df_struct = pl.struct(df_quantized_k.select(quant_columns))
    #
    # dfs_filtered = []
    # for lf in tqdm(lfs):
    #     lf_filtered = lf.filter(pl.struct(quant_columns).is_in(df_struct.implode()))
    #     df_filtered = lf_filtered.collect()
    #
    #     dfs_filtered.append(df_filtered)
    #
    # df = merge_quantized_pmw_features(dfs_filtered, quant_columns, AGG_OFF_COLUMNS)
    #
    # df_quantized.write_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "unique.parquet")
    # df.write_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "unique_k.parquet")
    #
    # # TODO: replace `unique` with `final` or `newest` for consistency
    # df_quantized = pl.read_parquet(pathlib.Path(DIR_PMW_ANALYSIS) / "unique.parquet")
    df_k = pl.read_parquet(df_dir_path / FILE_DF_FINAL_K)
    # df.select(["lon", "lat", "qualityFlag"]).row(distances.argmax())

    ###################
    # lf = pl.scan_parqueet(pathlib.Path(PMW_ANALYSIS_DIR) / "tmp" / "default" / "2018" / "1" / "final.parquet")
    # lf = lf.with_columns(pl.col("time").list.min().alias("time_min"))
    # lf = lf.sort("time_min", descending=True)
    #
    #
    # lf = lf.with_columns(pl.col(COLUMN_COUNT).cum_sum().alias(column_count_cumsum))
    # df = lf.filter(pl.col(column_count_cumsum) <= K).collect(engine="streaming")
    ###################

    m_occurrences = 1

    df_k_quant_m, quant_columns_with_suffix = filter_by_signature_occurrences_count(df_k, m_occurrences, quant_columns)

    df_k_m = df_k.join(df_k_quant_m, on=quant_columns_with_suffix, how="inner")
    df_k_m = df_k_m[quant_columns + [COLUMN_LON, COLUMN_LAT, "L1CqualityFlag", "qualityFlag"]]
    # df_k_m = df_k_m[[COLUMN_LON, COLUMN_LAT, "L1CqualityFlag"]]

    m_occurrences_text = "" if m_occurrences == 1 else f"; Signature occurred at least {m_occurrences} times."
    plot_variables_on_map(df_k_m, arg_transform,
                          images_dir=images_dir,
                          title_text_suffix=m_occurrences_text,
                          file_name_suffix=f"_{m_occurrences}")


if __name__ == '__main__':
    main()
