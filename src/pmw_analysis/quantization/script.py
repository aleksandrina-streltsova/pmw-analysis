"""
Script for performing quantization on data from bucket.
"""
import argparse
import logging
import pathlib
from typing import Dict, Callable, Iterable, List, Optional

import gpm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from tqdm import tqdm

from pmw_analysis.constants import BUCKET_DIR, PMW_ANALYSIS_DIR, COLUMN_LON, COLUMN_LAT, TC_COLUMNS, COLUMN_COUNT, \
    COLUMN_TIME
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict, quantize_pmw_features, \
    merge_quantized_pmw_features
from pmw_analysis.utils.logging import disable_logging, timing
from pmw_analysis.utils.polars import weighted_quantiles

QUANTIZE_MEMORY_USAGE_FACTOR = 10
MERGE_MEMORY_USAGE_FACTOR = 50
MEMORY_USAGE_LIMIT = 800

UNCERTAINTY_FACTOR_MAX = 20

X_STEP = 10
Y_STEP = 4


def _quantize_with_memory_check(df: pl.DataFrame, quant_columns: Iterable[str],
                                unc_dict: Dict[str, float], range_dict: Optional[Dict[str, float]],
                                limit: float) -> pl.DataFrame:
    estimated_usage = QUANTIZE_MEMORY_USAGE_FACTOR * df.estimated_size("gb")
    if estimated_usage > limit:
        raise MemoryError(
            f"Quantization aborted: estimated memory usage ({estimated_usage:.2f} GB) "
            f"exceeds limit ({limit:.2f} GB)."
        )

    return quantize_pmw_features(df, quant_columns, unc_dict, range_dict)


def _merge_with_memory_check(dfs: Iterable[pl.DataFrame], quant_columns: List[str], limit: float):
    estimated_size = sum(df.estimated_size("gb") for df in dfs)
    estimated_usage = MERGE_MEMORY_USAGE_FACTOR * estimated_size

    if estimated_usage > limit:
        raise MemoryError(
            f"Merging aborted: estimated memory usage ({estimated_usage:.2f} GB) "
            f"exceeds limit ({limit:.2f} GB)."
        )
    return merge_quantized_pmw_features(dfs, quant_columns)


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
    """
    Quantize fractions of data from bucket.
    """
    with open(path / "factor", "r", encoding="utf-8") as file:
        factor = float(file.read())
    unc_dict = {col: factor * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}
    range_dict = _get_ranges_dict(["default", "pd", "ratio"], plot_hists=False)
    quant_columns = transform(TC_COLUMNS)

    x_bounds, y_bounds = _calculate_bounds()
    for idx_x, (x_bound_l, x_bound_r) in enumerate(zip(x_bounds, x_bounds[1:])):
        for idx_y, (y_bound_l, y_bound_r) in tqdm(enumerate(zip(y_bounds, y_bounds[1:])), total=len(y_bounds) - 1):
            idx = idx_x * (len(y_bounds) - 1) + idx_y

            path_single = path / f"{idx}.parquet"
            if path_single.exists():
                continue

            extent = [x_bound_l, x_bound_r, y_bound_l, y_bound_r]

            with timing("Reading from bucket"):
                df: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR, columns=None, extent=extent)
            df = transform(df)

            with timing("Quantizing"):
                df_result = _quantize_with_memory_check(df, quant_columns, unc_dict, range_dict, MEMORY_USAGE_LIMIT)

            with timing("Writing"):
                df_result.write_parquet(path_single)

            logging.info(len(df) / len(df_result))


def _merge_partial(path: pathlib.Path, path_next: pathlib.Path, path_final: pathlib.Path, n: int, transform: Callable):
    quant_columns = transform(TC_COLUMNS)
    n_processed = 0

    idx_merged = 0
    while n_processed < n:
        estimated_usage = 0
        dfs = []
        while n_processed < n:
            df = pl.read_parquet(path / f"{n_processed}.parquet")
            estimated_usage += df.estimated_size("gb") * MERGE_MEMORY_USAGE_FACTOR
            if estimated_usage > MEMORY_USAGE_LIMIT:
                logging.info("%s: merging %d data frames", path_next.name, len(dfs))
                break
            dfs.append(df)
            n_processed += 1
        df_merged = _merge_with_memory_check(dfs, quant_columns, MEMORY_USAGE_LIMIT)
        if len(dfs) == n:
            df_merged.write_parquet(path_final / "final.parquet")
        else:
            df_merged.write_parquet(path_next / f"{idx_merged}.parquet")
        idx_merged += 1

    return idx_merged


def merge(path: pathlib.Path, transform: Callable):
    """
    Merge quantized fractions of data from bucket.
    """
    if (path / "final.parquet").exists():
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

        n = _merge_partial(path, path_next, path_final, n, transform)
        path = path_next

        if n == 1:
            break


def pd_transform(obj):
    """
    Replace vertical polarizations with polarization differences when possible.
    """
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

    if isinstance(obj, List):
        return obj

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def ratio_transform(obj):
    """
    Divide values by the values of 19H.
    """
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
            unc_dict[tc_col] = (unc_dict[tc_col] + unc_dict[tc_denom]) / 100
        return unc_dict

    if isinstance(obj, List):
        return obj

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v1_transform(obj):
    if isinstance(obj, pl.DataFrame):
        df = obj
        df = df.with_columns(
            pl.col("Tc_37H").truediv(pl.col("Tc_19H")).alias("Tc_37H_Tc_19H"),
            pl.col(f"Tc_89V").sub(pl.col(f"Tc_89H")).alias("PD_89"),
        )
        df = df.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V7"]])
        return df

    if isinstance(obj, Dict):
        unc_dict = obj
        unc_dict["Tc_37H_Tc_19H"] = (unc_dict["Tc_37H"] + unc_dict["Tc_19H"]) / 100
        unc_dict["PD_89"] = (unc_dict["Tc_89V"] + unc_dict["Tc_89H"]) / 2
        return unc_dict

    if isinstance(obj, List):
        return ["Tc_37H_Tc_19H", "PD_89", "Tc_23V", "Tc_165V", "Tc_183V7"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def estimate_uncertainty_factor(path: pathlib.Path, transform: Callable):
    """
    Estimate a factor which uncertainty should be multiplied by during quantization.
    Write the estimated factor into a file in the parent directory of the specified path.
    """
    unc_dict = transform(get_uncertainties_dict(TC_COLUMNS))
    # TODO: rewrite to use transform
    range_dict = _get_ranges_dict(["default", "pd", "ratio"], plot_hists=False)

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
        df: pl.DataFrame = gpm.bucket.read(BUCKET_DIR, extent, columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME])
        dfs.append(transform(df))

    quant_columns = transform(TC_COLUMNS)

    uncertainty_factor_l = 1
    uncertainty_factor_r = UNCERTAINTY_FACTOR_MAX

    while uncertainty_factor_r > uncertainty_factor_l + 1:
        factor = (uncertainty_factor_l + uncertainty_factor_r + 1) // 2
        logging.info("Checking uncertainty factor %d", factor)

        sizes_before = []
        sizes_after = []

        for df in tqdm(dfs):
            curr_uns_dict = {k: factor * v for k, v in unc_dict.items()}
            with disable_logging():
                df_quantized = quantize_pmw_features(df, quant_columns, curr_uns_dict, range_dict)

            sizes_before.append(len(df))
            sizes_after.append(len(df_quantized))
        before_to_after_ratio = np.median(np.array(sizes_before) / np.array(sizes_after))
        logging.info("before/after: %.2f", before_to_after_ratio)
        if before_to_after_ratio < 100:
            uncertainty_factor_l = factor
        else:
            uncertainty_factor_r = factor
    logging.basicConfig(level=logging.INFO)

    factor = uncertainty_factor_r
    logging.info("Final uncertainty factor: %d", factor)
    with open(path / "factor", "w", encoding="utf-8") as file:
        file.write(str(factor))

    return factor


def _get_ranges_dict(dirs: List[str], plot_hists: bool = False) -> Dict[str, float]:
    ranges_dict = {}

    for dir in dirs:
        df_path = pathlib.Path(PMW_ANALYSIS_DIR) / dir / "final.parquet"
        df_merged: pl.DataFrame = pl.read_parquet(df_path)
        df_merged = df_merged.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
        df_merged = df_merged.select(TC_COLUMNS + [COLUMN_COUNT])

        if dir == "pd":
            df_merged = df_merged.rename({f"Tc_{freq}V": f"PD_{freq}" for freq in [19, 37, 89, 165]})
            df_merged = df_merged.drop([tc_col for tc_col in TC_COLUMNS if tc_col in df_merged.columns])
        elif dir == "ratio":
            tc_denom = "Tc_19H"
            df_merged = df_merged.rename(
                {tc_col: f"{tc_col}_{tc_denom}" for tc_col in TC_COLUMNS if tc_col != tc_denom})
            df_merged = df_merged.drop([tc_col for tc_col in TC_COLUMNS if tc_col in df_merged.columns])

        for tc_col in [col for col in df_merged.columns if col != COLUMN_COUNT]:
            value_count = df_merged[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
            value_count = value_count.filter(pl.col(tc_col).is_not_null())
            value_count = value_count.sort(tc_col)

            q_min, q_max = weighted_quantiles(value_count, [1e-6, 1 - 1e-6], tc_col, COLUMN_COUNT)
            ranges_dict[f"{tc_col}_min"] = q_min
            ranges_dict[f"{tc_col}_max"] = q_max

            if plot_hists:
                fig, ax = plt.subplots(figsize=(10, 6))
                width = np.median((value_count[tc_col][1:] - value_count[tc_col][:-1]).to_numpy()) / 2
                ax.bar(value_count[tc_col], value_count[COLUMN_COUNT], width=width)
                ax.set_xlabel(tc_col)
                ax.set_ylabel(COLUMN_COUNT)
                ax.set_yscale("log")
                ax.set_title(f"{tc_col}")
                fig.tight_layout()

                y_min, y_max = ax.get_ylim()
                ax.vlines([q_min, q_max], ymin=y_min, ymax=y_max, colors="r")

                fig.show()
    return ranges_dict


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run quantization")

    parser.add_argument("--step", "-s", default="factor", choices=["factor", "quantize", "merge"],
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", "-t", default="default", choices=["default", "pd", "ratio", "v1"],
                        help="Type of transformation to perform on data")

    args = parser.parse_args()

    path = pathlib.Path(PMW_ANALYSIS_DIR) / args.transform
    path.mkdir(parents=True, exist_ok=True)

    if args.transform == "pd":
        transform = pd_transform
    elif args.transform == "ratio":
        transform = ratio_transform
    elif args.transform == "v1":
        transform = v1_transform
    else:
        transform = lambda x: x

    if args.step == "factor":
        estimate_uncertainty_factor(path, transform)
    elif args.step == "quantize":
        quantize(path, transform)
    else:
        merge(path, transform)


if __name__ == '__main__':
    main()
