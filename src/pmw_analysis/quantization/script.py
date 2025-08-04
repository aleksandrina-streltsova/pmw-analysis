"""
Script for performing quantization on data from bucket.
"""
import logging
import pathlib
from typing import Dict, Callable, List, Sequence

import configargparse
import gpm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from tqdm import tqdm

from pmw_analysis.constants import BUCKET_DIR, PMW_ANALYSIS_DIR, COLUMN_LON, COLUMN_LAT, TC_COLUMNS, COLUMN_COUNT, \
    COLUMN_TIME, DEBUG_FLAG, COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID, COLUMN_LON_BIN, COLUMN_LAT_BIN, \
    COLUMN_SUFFIX_QUANT, COLUMN_OCCURRENCE
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict, quantize_pmw_features, \
    merge_quantized_pmw_features
from pmw_analysis.utils.logging import disable_logging, timing
from pmw_analysis.utils.polars import weighted_quantiles, take_k_sorted

UNCERTAINTY_FACTOR_MAX = 20

X_STEP = 10
Y_STEP = 4

K = 100000


def _calculate_bounds():
    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)
    x_bounds = p.x_bounds.tolist()
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    x_include_final = len(x_bounds) % X_STEP == 0
    y_include_final = len(y_bounds) % Y_STEP == 0

    x_bounds = x_bounds[::X_STEP] + ([x_bounds[-1]] if x_include_final else [])
    y_bounds = y_bounds[::Y_STEP] + ([y_bounds[-1]] if y_include_final else [])

    return x_bounds, y_bounds


def quantize(path: pathlib.Path, transform: Callable, factor: float,
             month: int | None,
             year: int | None,
             agg_off_columns: List[str]):
    """
    Quantize fractions of data from bucket.
    """
    unc_dict = {col: factor * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}
    range_dict = _get_ranges_dict(["default", "pd", "ratio"], plot_hists=False)
    quant_columns = transform(TC_COLUMNS)

    x_bounds, y_bounds = _calculate_bounds()
    for idx_x, (x_bound_l, x_bound_r) in enumerate(zip(x_bounds, x_bounds[1:])):
        for idx_y, (y_bound_l, y_bound_r) in tqdm(enumerate(zip(y_bounds, y_bounds[1:])), total=1):
            idx = idx_x * (len(y_bounds) - 1) + idx_y

            path_single = path / f"{idx}.parquet"
            if path_single.exists():
                continue

            extent = [x_bound_l, x_bound_r, y_bound_l, y_bound_r]

            with timing("Reading from bucket"):
                lf: pl.LazyFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR, columns=None, extent=extent,
                                                   backend="polars_lazy")
                if year is not None:
                    lf = lf.filter(pl.col(COLUMN_TIME).dt.year() == year)
                if month is not None:
                    lf = lf.filter(pl.col(COLUMN_TIME).dt.month() == month)

            lf = transform(lf)

            with timing("Quantizing"):
                lf_result = quantize_pmw_features(lf, quant_columns, unc_dict, range_dict, agg_off_columns)

            with timing("Writing"):
                lf_result.sink_parquet(path_single, engine="streaming")

            if DEBUG_FLAG:
                logging.info(lf.select(pl.len()).collect(engine="streaming").item() /
                             lf_result.select(pl.len()).collect(engine="streaming").item())


def _merge_partial(path: pathlib.Path, path_next: pathlib.Path, path_final: pathlib.Path,
                   n: int, transform: Callable, agg_off_columns: List[str]):
    quant_columns = transform(TC_COLUMNS)
    n_processed = 0

    idx_merged = 0
    while n_processed < n:
        # estimated_usage = 0
        lfs = []
        while n_processed < n:
            lf = pl.scan_parquet(path / f"{n_processed}.parquet")
            # TODO: estimate size of lazy frame
            # estimated_usage += df.estimated_size("gb") * MERGE_MEMORY_USAGE_FACTOR
            # if estimated_usage > MEMORY_USAGE_LIMIT:
            #     logging.info("%s: merging %d data frames", path_next.name, len(dfs))
            #     break
            if len(lfs) == 2:
                logging.info("%s: merging %d data frames", path_next.name, len(lfs))
                break
            lfs.append(lf)
            n_processed += 1
        lf_merged = merge_quantized_pmw_features(lfs, quant_columns, agg_off_columns)
        if len(lfs) == n:
            lf_merged.sink_parquet(path_final / "final.parquet", engine="streaming")
        else:
            lf_merged.sink_parquet(path_next / f"{idx_merged}.parquet", engine="streaming")
        idx_merged += 1

    return idx_merged


def merge(path: pathlib.Path, transform: Callable, agg_off_columns: List[str]):
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

        n = _merge_partial(path, path_next, path_final, n, transform, agg_off_columns)
        path = path_next

        if n == 1:
            break


def pd_transform(obj):
    """
    Replace vertical polarizations with polarization differences when possible.
    """
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        for freq in [19, 37, 89, 165]:
            lf = lf.with_columns(pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")))
        lf = lf.with_columns(pl.col("Tc_183V7").sub(pl.col("Tc_183V3")))
        return lf

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

    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        for tc_col in TC_COLUMNS:
            if tc_col == tc_denom:
                continue
            lf = lf.with_columns(pl.col(tc_col).truediv(pl.col(tc_denom)))
        return lf

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
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns(
            pl.col("Tc_37H").truediv(pl.col("Tc_19H")).alias("Tc_37H_Tc_19H"),
            pl.col("Tc_89V").sub(pl.col("Tc_89H")).alias("PD_89"),
        )
        lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V7"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        unc_dict["Tc_37H_Tc_19H"] = (unc_dict["Tc_37H"] + unc_dict["Tc_19H"]) / 100
        unc_dict["PD_89"] = (unc_dict["Tc_89V"] + unc_dict["Tc_89H"]) / 2
        return unc_dict

    if isinstance(obj, List):
        return ["Tc_37H_Tc_19H", "PD_89", "Tc_23V", "Tc_165V", "Tc_183V7"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v2_transform(obj):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns(
            pl.col("Tc_37H").truediv(pl.col("Tc_19H")).alias("Tc_37H_Tc_19H"),
            pl.col("Tc_89V").sub(pl.col("Tc_89H")).alias("PD_89"),
        )
        lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_19V", "Tc_89V"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        unc_dict["Tc_37H_Tc_19H"] = (unc_dict["Tc_37H"] + unc_dict["Tc_19H"]) / 100
        unc_dict["PD_89"] = (unc_dict["Tc_89V"] + unc_dict["Tc_89H"]) / 2
        return unc_dict

    if isinstance(obj, List):
        return ["Tc_37H_Tc_19H", "PD_89", "Tc_19V", "Tc_89V"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v3_transform(obj):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns(
            pl.col("Tc_165V").sub(pl.col("Tc_165H")).alias("PD_165"),
            pl.col("Tc_183V3").sub(pl.col("Tc_183V7")).alias("PD_183"),
        )
        lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V3"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        unc_dict["PD_165"] = (unc_dict["Tc_165V"] + unc_dict["Tc_165H"]) / 2
        unc_dict["PD_183"] = (unc_dict["Tc_183V3"] + unc_dict["Tc_183V7"]) / 2
        return unc_dict

    if isinstance(obj, List):
        return ["Tc_23V", "Tc_165V", "PD_165", "Tc_183V3", "PD_183"]

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

    lfs = []
    k = 15
    for idx_x, idx_y in tqdm(zip(np.random.choice(np.arange(len(x_bounds) - 1), k),
                                 np.random.choice(np.arange(len(y_bounds) - 1), k)), total=k):
        extent = [x_bounds[idx_x], x_bounds[idx_x + 1], y_bounds[idx_y], y_bounds[idx_y + 1]]
        lf: pl.LazyFrame = gpm.bucket.read(BUCKET_DIR, extent,
                                           columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME],
                                           backend="polars_lazy")
        lfs.append(transform(lf))

    quant_columns = transform(TC_COLUMNS)

    factor = _find_factor_using_binary_search(lfs, quant_columns, unc_dict, range_dict)

    logging.basicConfig(level=logging.INFO)
    logging.info("Final uncertainty factor: %d", factor)
    with open(path / "factor", "w", encoding="utf-8") as file:
        file.write(str(factor))

    return factor


def _find_factor_using_binary_search(lfs: Sequence[pl.DataFrame | pl.LazyFrame], quant_columns: Sequence[str],
                                     uncertainty_dict: Dict[str, float], range_dict: Dict[str, float]):
    uncertainty_factor_l = 0
    uncertainty_factor_r = UNCERTAINTY_FACTOR_MAX
    while uncertainty_factor_r > uncertainty_factor_l + 1:
        factor = (uncertainty_factor_l + uncertainty_factor_r + 1) // 2
        logging.info("Checking uncertainty factor %d", factor)

        sizes_before = []
        sizes_after = []

        for lf in tqdm(lfs):
            curr_uns_dict = {k: factor * v for k, v in uncertainty_dict.items()}
            with disable_logging():
                lf_quantized = quantize_pmw_features(lf, quant_columns, curr_uns_dict, range_dict)

            sizes_before.append(lf.select(pl.len()).collect(engine="streaming").item())
            sizes_after.append(lf_quantized.select(pl.len()).collect(engine="streaming").item())
        before_to_after_ratio = np.median(np.array(sizes_before) / np.array(sizes_after))
        logging.info("before/after: %.2f", before_to_after_ratio)
        if before_to_after_ratio < 100:
            uncertainty_factor_l = factor
        else:
            uncertainty_factor_r = factor
    return uncertainty_factor_r


def _get_ranges_dict(dir_names: List[str], plot_hists: bool = False) -> Dict[str, float]:
    ranges_dict = {}

    for dir_name in dir_names:
        df_path = pathlib.Path(PMW_ANALYSIS_DIR) / dir_name / "final.parquet"
        df_merged: pl.DataFrame = pl.read_parquet(df_path)
        df_merged = df_merged.with_columns(pl.col(COLUMN_COUNT).cast(pl.UInt64))
        df_merged = df_merged.select(TC_COLUMNS + [COLUMN_COUNT])

        if dir_name == "pd":
            df_merged = df_merged.rename({f"Tc_{freq}V": f"PD_{freq}" for freq in [19, 37, 89, 165]})
            df_merged = df_merged.drop([tc_col for tc_col in TC_COLUMNS if tc_col in df_merged.columns])
        elif dir_name == "ratio":
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
    # TODO: fix
    ranges_dict["PD_183_min"] = -np.inf
    ranges_dict["PD_183_max"] = np.inf
    return ranges_dict


def get_newest_k(path: pathlib.Path, k: int):
    # TODO: extract file name to constant
    df_id = pl.read_parquet(path / "final.parquet")
    df_id_k =  take_k_sorted(df_id, COLUMN_OCCURRENCE, k, COLUMN_COUNT, descending=True)
    df_id_k.write_parquet(path / "final_k.parquet")


def _get_bucket_data_for_ids(df_id_k: pl.DataFrame) -> pl.DataFrame:
    id_columns = [COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID]

    p: LonLatPartitioning = get_bucket_spatial_partitioning(BUCKET_DIR)

    df_id_k = df_id_k.explode(p.levels + id_columns)

    df_id_k = p.add_labels(df_id_k, x=COLUMN_LON, y=COLUMN_LAT)
    df_id_k = p.add_centroids(df_id_k, x=COLUMN_LON, y=COLUMN_LAT, x_coord=COLUMN_LON_BIN, y_coord=COLUMN_LAT_BIN)

    df_id_k_grouped = df_id_k.group_by(p.levels)
    agg_off_columns = [col for col in df_id_k.columns if col not in p.levels]
    df_id_k_agg = df_id_k_grouped.agg(pl.col(agg_off_columns)).sort(p.levels)

    dfs_k_bin = []

    for x_min, x_max, x_c in tqdm(zip(p.x_bounds[:-1], p.x_bounds[1:], p.x_centroids), total=len(p.x_centroids)):
        for y_min, y_max, y_c in zip(p.y_bounds[:-1], p.y_bounds[1:], p.y_centroids):
            df_k_bin = df_id_k_agg.filter(pl.col(COLUMN_LON_BIN) == x_c, pl.col(COLUMN_LAT_BIN) == y_c).drop(p.levels)
            df_k_bin = df_k_bin.explode(agg_off_columns)

            if len(df_k_bin) == 0:
                continue

            extent = [x_min, x_max, y_min, y_max]
            df_bin = gpm.bucket.read(bucket_dir=BUCKET_DIR,
                                     extent=extent,
                                     backend="polars")
            df_k_bin = df_k_bin.join(df_bin, on=id_columns, how="inner", suffix=COLUMN_SUFFIX_QUANT)
            dfs_k_bin.append(df_k_bin)

    df_k = pl.concat(dfs_k_bin)
    return df_k


def get_transformation_function(arg_transform: str) -> Callable:
    """
    Return a transformation function based on the specified argument.
    """
    if arg_transform == "pd":
        transform = pd_transform
    elif arg_transform == "ratio":
        transform = ratio_transform
    elif arg_transform == "v1":
        transform = v1_transform
    elif arg_transform == "v2":
        transform = v2_transform
    elif arg_transform == "v3":
        transform = v3_transform
    else:
        transform = lambda x: x

    return transform


def main():
    logging.basicConfig(level=logging.INFO)

    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run quantization")

    parser.add_argument("--step", default="factor", choices=["factor", "quantize", "merge", "newest-k"],
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", default="default",
                        choices=["default", "pd", "ratio", "v1", "v2", "v3"],
                        help="Type of transformation to perform on data")
    parser.add_argument("--dir", default=PMW_ANALYSIS_DIR,
                        help="Path to the directory to store quantized data in")
    parser.add_argument("--agg-off-cols", default=[], nargs="+",
                        help="Columns whose values are stored in lists, without aggregation")
    parser.add_argument("--month", type=int, help="Month of the data to quantize")
    parser.add_argument("--year", type=int, help="Year of the data to quantize")

    args = parser.parse_args()

    assert not (args.year is None and args.month is not None)

    path = pathlib.Path(args.dir) / args.transform
    path.mkdir(parents=True, exist_ok=True)

    transform = get_transformation_function(args.transform)
    if args.step == "factor":
        estimate_uncertainty_factor(path, transform)
        return

    with open(path / "factor", "r", encoding="utf-8") as file:
        factor = float(file.read())

    if args.year is not None:
        path = path / str(args.year)
    if args.month is not None:
        path = path / str(args.month)
    path.mkdir(parents=True, exist_ok=True)

    if args.step == "quantize":
        quantize(path, transform, factor, args.month, args.year, args.agg_off_cols)
    elif args.step == "merge":
        merge(path, transform, args.agg_off_cols)
    else:
        get_newest_k(path, K)



if __name__ == '__main__':
    main()
