import argparse
import logging
import pathlib
from typing import Dict, Callable, Iterable

import gpm
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from tqdm import tqdm

from pmw_analysis.constants import BUCKET_DIR, PMW_ANALYSIS_DIR, COLUMN_LON, COLUMN_LAT, TC_COLUMNS
from pmw_analysis.preprocessing_polars import get_uncertainties_dict, quantize_pmw_features, \
    merge_quantized_pmw_features, quantize_features
from pmw_analysis.utils.logging import disable_logging, timing

QUANTIZE_MEMORY_USAGE_FACTOR = 10
MERGE_MEMORY_USAGE_FACTOR = 50
MEMORY_USAGE_LIMIT = 800

UNCERTAINTY_FACTOR_MAX = 20

X_STEP = 10
Y_STEP = 4

def _quantize_with_memory_check(df: pl.DataFrame, unc_dict: Dict[str, float], limit: float):
    estimated_usage = QUANTIZE_MEMORY_USAGE_FACTOR * df.estimated_size("gb")
    if estimated_usage > limit:
        raise MemoryError(
            f"Quantization aborted: estimated memory usage ({estimated_usage:.2f} GB) "
            f"exceeds limit ({limit:.2f} GB)."
        )

    return quantize_pmw_features(df, unc_dict)


def _merge_with_memory_check(dfs: Iterable[pl.DataFrame], limit: float):
    estimated_size = sum(df.estimated_size("gb") for df in dfs)
    estimated_usage = MERGE_MEMORY_USAGE_FACTOR * estimated_size

    if estimated_usage > limit:
        raise MemoryError(
            f"Merging aborted: estimated memory usage ({estimated_usage:.2f} GB) "
            f"exceeds limit ({limit:.2f} GB)."
        )
    return merge_quantized_pmw_features(dfs)


def _calculate_bounds():
    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds.tolist()
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    x_include_final = len(x_bounds) % X_STEP == 0
    y_include_final = len(y_bounds) % Y_STEP == 0

    x_bounds = x_bounds[::X_STEP] + ([x_bounds[-1]] if x_include_final else [])
    y_bounds = y_bounds[::Y_STEP] + ([y_bounds[-1]] if y_include_final else [])

    return x_bounds, y_bounds


def quantize(path: pathlib.Path, transform: Callable):
    with open(path / "factor", "r") as file:
        factor = float(file.read())

    x_bounds, y_bounds = _calculate_bounds()
    for idx_x, (x_bound_left, x_bound_right) in enumerate(zip(x_bounds, x_bounds[1:])):
        for idx_y, (y_bound_left, y_bound_right) in tqdm(enumerate(zip(y_bounds, y_bounds[1:])), total=len(y_bounds) - 1):
            idx = idx_x * (len(y_bounds) - 1) + idx_y

            path_single = path / f"{idx}.parquet"
            if path_single.exists():
                continue

            extent = [x_bound_left, x_bound_right, y_bound_left, y_bound_right]

            with timing(f"Reading from bucket"):
                df: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR, columns=None, extent=extent)

            df = transform(df)
            unc_dict = {col: factor * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}

            with timing(f"Quantizing"):
                df_result = _quantize_with_memory_check(df, unc_dict, MEMORY_USAGE_LIMIT)

            with timing(f"Writing"):
                df_result.write_parquet(path_single)

            logging.info(len(df) / len(df_result))


def _merge_partial(path: pathlib.Path, path_next: pathlib.Path, path_final: pathlib.Path, n: int):
    n_processed = 0

    idx_merged = 0
    while n_processed < n:
        estimated_usage = 0
        dfs = []
        while n_processed < n:
            df = pl.read_parquet(path / f"{n_processed}.parquet")
            estimated_usage += df.estimated_size("gb") * MERGE_MEMORY_USAGE_FACTOR
            if estimated_usage > MEMORY_USAGE_LIMIT:
                logging.info(f"{path_next.name}: merging {len(dfs)} data frames")
                break
            dfs.append(df)
            n_processed += 1
        df_merged = _merge_with_memory_check(dfs, MEMORY_USAGE_LIMIT)
        if len(dfs) == n:
            df_merged.write_parquet(path_final / f"final.parquet")
        else:
            df_merged.write_parquet(path_next / f"{idx_merged}.parquet")
        idx_merged += 1

    return idx_merged


def merge(path: pathlib.Path):
    if (path / f"final.parquet").exists():
        return

    x_bounds, y_bounds = _calculate_bounds()
    n = (len(x_bounds) - 1) * (len(y_bounds) - 1)

    level = 0
    path_final = path
    while True:
        path_next = path / f"level{level}"
        path_next.mkdir(exist_ok=True)

        level += 1
        if (path_next / f"level{level}").exists():
            n = sum(1 for f in path_next.iterdir() if f.is_file())
            path = path_next
            continue

        n = _merge_partial(path, path_next, path_final, n)
        path = path_next

        if n == 1:
            break


def PD_transform(obj):
    if isinstance(obj, pl.DataFrame):
        df = obj
        for freq in [19, 37, 89, 165]:
            df = df.with_columns(pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")))
        df = df.with_columns(pl.col("Tc_183V7").sub(pl.col("Tc_183V3")))
        return df

    if isinstance(obj, Dict):
        unc_dict = obj
        for freq in [19, 37, 89, 165]:
            unc_dict[f"Tc_{freq}V"] = (unc_dict[f"Tc_{freq}H"] + unc_dict[f"Tc_{freq}V"]) / 2
        unc_dict["Tc_183V7"] = (unc_dict["Tc_183V3"] + unc_dict["Tc_183V7"]) / 2
        return unc_dict


def ratio_transform(obj):
    tc_denom = "Tc_19H"

    if isinstance(obj, pl.DataFrame):
        df = obj
        for tc_col in TC_COLUMNS:
            if tc_col == tc_denom:
                continue
            df = df.with_columns(pl.col(tc_col).truediv(pl.col(tc_denom)))
        return df

    if isinstance(obj, Dict):
        unc_dict = obj
        for tc_col in TC_COLUMNS:
            if tc_col == tc_denom:
                continue
            unc_dict[tc_col] = (unc_dict[tc_col] + unc_dict[tc_denom]) / 200
        return unc_dict


def get_uncertainty_factor(path: pathlib.Path, transform: Callable):
    logging.info("Reading data")
    np.random.seed(566)

    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    dfs = []
    k = 15
    for idx_x, idx_y in tqdm(zip(np.random.choice(np.arange(len(x_bounds) - 1), k),
                                 np.random.choice(np.arange(len(y_bounds) - 1), k)), total=k):
        extent = [x_bounds[idx_x], x_bounds[idx_x + 1], y_bounds[idx_y], y_bounds[idx_y + 1]]
        df: pl.DataFrame = gpm.bucket.read(BUCKET_DIR, extent, columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT])
        dfs.append(transform(df.select(TC_COLUMNS)))

    unc_dict = transform(get_uncertainties_dict(TC_COLUMNS))

    uncertainty_factor_l = 1
    uncertainty_factor_r = UNCERTAINTY_FACTOR_MAX

    while uncertainty_factor_r > uncertainty_factor_l + 1:
        factor = (uncertainty_factor_l + uncertainty_factor_r + 1) // 2
        logging.info(f"Checking uncertainty factor {factor}")

        sizes_before = []
        sizes_after = []

        for df in tqdm(dfs):
            with disable_logging():
                df_quantized = quantize_features(df, TC_COLUMNS, [factor * unc_dict[tc_col] for tc_col in TC_COLUMNS])

            sizes_before.append(len(df))
            sizes_after.append(len(df_quantized))

        if np.median(np.array(sizes_before) / np.array(sizes_after)) < 100:
            uncertainty_factor_l = factor
        else:
            uncertainty_factor_r = factor
    logging.basicConfig(level=logging.INFO)

    factor = uncertainty_factor_r
    logging.info(f"Final uncertainty factor: {factor}")
    with open(path / "factor", "w") as file:
        file.write(str(factor))

    return factor


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run quantization")

    parser.add_argument("--step", "-s", default="factor", choices=["factor", "quantize", "merge"],
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio"],
                        help="Type of transformation to perform on data")

    args = parser.parse_args()

    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform
    path.mkdir(parents=True, exist_ok=True)

    if args.transform == "pd":
        transform = PD_transform
    elif args.transform == "ratio":
        transform = ratio_transform
    else:
        transform = lambda x: x

    if args.step == "factor":
        factor = get_uncertainty_factor(path, transform)
    elif args.step == "quantize":
        quantize(path, transform)
    else:
        merge(path)
