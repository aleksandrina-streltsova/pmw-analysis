import logging
import pathlib
import sys
from typing import Dict

import gpm
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from tqdm import tqdm

from pmw_analysis.constants import BUCKET_DIR, PMW_ANALYSIS_DIR, COLUMN_LON, COLUMN_LAT
from pmw_analysis.preprocessing_polars import get_uncertainties_dict, quantize_pmw_features, \
    merge_quantized_pmw_features

N_THREADS = 6

def segment(thread_id: int, path: pathlib.Path):
    with open(path / "factor", "r") as file:
        factor = float(file.read())

    logging.info(str(factor))

    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    batch_size = len(x_bounds) // N_THREADS

    for idx_x, (x_bound_left, x_bound_right) in tqdm(enumerate(zip(x_bounds, x_bounds[1:]))):
        if idx_x < thread_id * batch_size or idx_x >= (thread_id + 1) * batch_size:
            continue

        for idx_y, (y_bound_left, y_bound_right) in enumerate(zip(y_bounds, y_bounds[1:])):
            if (path / f"{idx_x}_{idx_y}.parquet").exists():
                continue

            extent = [x_bound_left, x_bound_right, y_bound_left, y_bound_right]

            df: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR,
                                               columns=None,
                                               use_pyarrow=False,  # use rust parquet reader
                                               extent=extent,
                                               # n_rows=2,
                                               # n_rows=100_000_000, # for prototyping
                                               parallel="auto",  # "row_groups", "columns"
                                               )
            df = PD_transform(df)
            df_result = quantize_pmw_features(df, factor)
            df_result.write_parquet(path / f"{idx_x}_{idx_y}.parquet")

            logging.info(len(df) / len(df_result))


def merge(thread_id: int, path: pathlib.Path):
    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    dfs_merged = []

    batch_size = len(x_bounds) // N_THREADS
    for idx_x in tqdm(range(thread_id * batch_size, (thread_id + 1) * batch_size)):
        dfs = []
        for idx_y in range(len(y_bounds) - 1):
            df = pl.read_parquet(path / f"{idx_x}_{idx_y}.parquet")
            dfs.append(df)
        df_merged = merge_quantized_pmw_features(dfs)
        dfs_merged.append(df_merged)

    df_merged = merge_quantized_pmw_features(dfs_merged)
    df_merged.write_parquet(path / f"/{thread_id}.parquet")


def merge_final(path: pathlib.Path):
    dfs = []
    for idx in range(N_THREADS):
        df = pl.read_parquet(path / f"{idx}.parquet")
        dfs.append(df)
    df_final = merge_quantized_pmw_features(dfs)
    df_final.write_parquet(path / "final.parquet")


def PD_transform(obj):
    if isinstance(obj, pl.DataFrame):
        df = obj

        for freq in [10, 19, 37, 89, 165]:
            df = df.with_columns(pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")))
        df = df.with_columns(pl.col("Tc_183V7").sub(pl.col("Tc_183V3")))
        return df

    if isinstance(obj, Dict):
        unc_dict = obj
        for freq in [10, 19, 37, 89, 165]:
            unc_dict[f"Tc_{freq}V"] =  unc_dict[f"Tc_{freq}H"] +  unc_dict[f"Tc_{freq}V"]
        unc_dict["Tc_183V7"] = unc_dict["Tc_183V3"] + unc_dict["Tc_183V7"]
        return unc_dict


def get_uncertainty_factor(path: pathlib.Path):
    np.random.seed(566)

    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))
    tc_cols = ['Tc_10H', 'Tc_10V', 'Tc_19H', 'Tc_19V', 'Tc_23V', 'Tc_37H', 'Tc_37V', 'Tc_89H', 'Tc_89V', 'Tc_165H',
               'Tc_165V', 'Tc_183V3', 'Tc_183V7']
    dfs = []
    for idx_x, idx_y in zip(np.random.choice(np.arange(len(x_bounds) - 1), 5), np.random.choice(np.arange(len(y_bounds) - 1), 5)):
        extent = [x_bounds[idx_x], x_bounds[idx_x + 1], y_bounds[idx_y], y_bounds[idx_y + 1]]
        df: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR,
                                           columns=tc_cols + [COLUMN_LON, COLUMN_LAT],
                                           use_pyarrow=False,  # use rust parquet reader
                                           extent=extent,
                                           parallel="auto",  # "row_groups", "columns"
                                           )
        dfs.append(df)
    df = pl.concat(dfs)

    unc_dict = get_uncertainties_dict(tc_cols)

    df_transformed = PD_transform(df)
    unc_dict_transformed = PD_transform(unc_dict)

    before = np.zeros(len(tc_cols))
    after = np.zeros(len(tc_cols))
    for i, col in enumerate(tc_cols):
        before[i] = (df[col].max() - df[col].min()) / (10 * unc_dict[col])
        after[i] = (df_transformed[col].max() - df_transformed[col].min()) / unc_dict_transformed[col]

    factor =  np.power((after / before).prod(), 1 / len(tc_cols))

    with open(path / "factor", "w") as file:
        file.write(str(factor))

    return factor

# TODO: fix CLI
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    dir_name = sys.argv[2]

    path = pathlib.Path(PMW_ANALYSIS_DIR) / dir_name
    path.mkdir(parents=True, exist_ok=True)

    if sys.argv[1] == "factor":
        factor = get_uncertainty_factor(path)
        logging.info(f"uncertainty factor: {factor}")
    elif sys.argv[1] == "partial":
        thread_id = int(sys.argv[3])

        segment(thread_id, path)
        merge(thread_id, path)
    else:
        merge_final(path)


