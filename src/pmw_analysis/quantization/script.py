"""
Script for performing quantization on data from bucket.
"""
import logging
import multiprocessing
import pathlib
import pickle
from collections import defaultdict
from typing import Dict, Callable, List, Sequence, Tuple

import configargparse
import gpm
import numpy as np
import polars as pl
from gpm.bucket import LonLatPartitioning
from gpm.bucket.io import get_bucket_spatial_partitioning
from tqdm import tqdm

from pmw_analysis.constants import DIR_BUCKET, DIR_PMW_ANALYSIS, COLUMN_LON, COLUMN_LAT, TC_COLUMNS, COLUMN_COUNT, \
    COLUMN_TIME, FLAG_DEBUG, COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID, COLUMN_LON_BIN, COLUMN_LAT_BIN, \
    COLUMN_SUFFIX_QUANT, COLUMN_OCCURRENCE, FILE_DF_FINAL, FILE_DF_FINAL_K, \
    ArgQuantizationStep, ArgTransform, ArgQuantizationL2L3Columns, VARIABLE_SURFACE_TYPE_INDEX, COLUMN_L1C_QUALITY_FLAG, \
    DIR_NO_SUN_GLINT, ArgSurfaceType, COLUMN_BREAKPOINT, COLUMN_CATEGORY
from pmw_analysis.copypaste.utils.cli import EnumAction
from pmw_analysis.processing.filter import filter_by_surface_type
from pmw_analysis.quantization.dataframe_polars import get_uncertainties_dict, quantize_pmw_features, \
    merge_quantized_pmw_features, create_occurrence_column
from pmw_analysis.retrievals.retrieval_1b_c_pmw import retrieve_possible_sun_glint
from pmw_analysis.utils.io import rmtree
from pmw_analysis.utils.logging import disable_logging, timing, get_memory_usage
from pmw_analysis.utils.polars import take_k_sorted, weighted_quantiles

MERGE_MEMORY_USAGE_FACTOR = 50
MEMORY_USAGE_LIMIT = 800

UNCERTAINTY_FACTOR_MAX = 20

FLAG_TEST = False
N_DFS_TEST = 3

X_STEP = 10
Y_STEP = 4
X_STEP_TEST = 1
Y_STEP_TEST = 1

K = 100000


def _calculate_bounds(x_step: int = X_STEP, y_step: int = Y_STEP) -> Tuple[List[float], List[float]]:
    if FLAG_TEST:
        x_step = X_STEP_TEST
        y_step = Y_STEP_TEST

    p: LonLatPartitioning = get_bucket_spatial_partitioning(DIR_BUCKET)
    x_bounds = p.x_bounds.tolist()
    y_bounds = list(filter(lambda b: abs(b) <= 70, p.y_bounds))

    x_include_final = (len(x_bounds) - 1) % x_step != 0
    y_include_final = (len(y_bounds) - 1) % y_step != 0

    x_bounds = x_bounds[::x_step] + ([x_bounds[-1]] if x_include_final else [])
    y_bounds = y_bounds[::y_step] + ([y_bounds[-1]] if y_include_final else [])

    return x_bounds, y_bounds


def quantize(path: pathlib.Path, transform: Callable, filter_rows: Callable, factor: float, clip: bool,
             month: int | None,
             year: int | None,
             agg_off_columns: List[str],
             l2_l3_columns: ArgQuantizationL2L3Columns):
    """
    Quantize fractions of data from bucket.
    """
    unc_dict = {col: factor * unc for col, unc in transform(get_uncertainties_dict(TC_COLUMNS)).items()}
    range_dict = get_range_dict(clip)
    quant_columns = transform(TC_COLUMNS)

    x_bounds, y_bounds = _calculate_bounds()

    progress_bar = tqdm(total=(len(x_bounds) - 1) * (len(y_bounds) - 1))
    for idx_x, (x_bound_l, x_bound_r) in enumerate(zip(x_bounds, x_bounds[1:])):
        for idx_y, (y_bound_l, y_bound_r) in enumerate(zip(y_bounds, y_bounds[1:])):
            idx = idx_x * (len(y_bounds) - 1) + idx_y
            if FLAG_TEST and idx >= N_DFS_TEST:
                break

            path_single = path / f"{idx}.parquet"
            if path_single.exists():
                continue

            extent = [x_bound_l, x_bound_r, y_bound_l, y_bound_r]

            with timing("Reading from bucket"):
                occurrence_columns = [COLUMN_LON, COLUMN_LAT, COLUMN_TIME]

                required_columns = set(TC_COLUMNS + agg_off_columns + occurrence_columns)
                match l2_l3_columns:
                    case ArgQuantizationL2L3Columns.NONE:
                        columns = required_columns
                    case ArgQuantizationL2L3Columns.ANALYSIS_MINIMUM:
                        columns = required_columns
                        columns.add(VARIABLE_SURFACE_TYPE_INDEX)
                        columns.add(COLUMN_L1C_QUALITY_FLAG)
                    case ArgQuantizationL2L3Columns.ALL:
                        columns = None

                lf: pl.DataFrame = gpm.bucket.read(bucket_dir=DIR_BUCKET, columns=columns, extent=extent,
                                                   backend="polars")
                if year is not None:
                    lf = lf.filter(pl.col(COLUMN_TIME).dt.year() == year)
                if month is not None:
                    lf = lf.filter(pl.col(COLUMN_TIME).dt.month() == month)

            lf = transform(filter_rows(lf))

            with timing("Quantizing"):
                lf_result = quantize_pmw_features(lf, quant_columns, unc_dict, range_dict, agg_off_columns)

            with timing("Writing"):
                # lf_result.sink_parquet(path_single, engine="streaming")
                lf_result.write_parquet(path_single)

            if FLAG_DEBUG:
                logging.info(lf.select(pl.len()).collect(engine="streaming").item() /
                             lf_result.select(pl.len()).collect(engine="streaming").item())

            progress_bar.update(1)


def _merge_partial(path: pathlib.Path, path_next: pathlib.Path, path_final: pathlib.Path,
                   n: int, transform: Callable, agg_off_columns: List[str]):
    quant_columns = transform(TC_COLUMNS)
    n_processed = 0

    idx_merged = 0
    while n_processed < n:
        estimated_usage = 0
        lfs = []
        while n_processed < n:
            lf = pl.read_parquet(path / f"{n_processed}.parquet")
            # TODO: estimate size of lazy frame
            estimated_usage += lf.estimated_size("gb") * MERGE_MEMORY_USAGE_FACTOR
            if estimated_usage > MEMORY_USAGE_LIMIT:
                logging.info("%s: merging %d data frames", path_next.name, len(lfs))
                break
            # if len(lfs) == 4:
            #     logging.info("%s: merging %d data frames", path_next.name, len(lfs))
            #     break
            lfs.append(lf)
            n_processed += 1
        lf_merged = merge_quantized_pmw_features(lfs, quant_columns, agg_off_columns)
        if len(lfs) == n:
            # lf_merged.sink_parquet(path_final / "final.parquet", engine="streaming")
            lf_merged.write_parquet(path_final / "final.parquet")
        else:
            # lf_merged.sink_parquet(path_next / f"{idx_merged}.parquet", engine="streaming")
            lf_merged.write_parquet(path_next / f"{idx_merged}.parquet")
        idx_merged += 1

    return idx_merged


def merge(path: pathlib.Path, transform: Callable, agg_off_columns: List[str]):
    """
    Merge quantized fractions of data from bucket.
    """
    if (path / "final.parquet").exists():
        return

    x_bounds, y_bounds = _calculate_bounds()
    n = (len(x_bounds) - 1) * (len(y_bounds) - 1) if not FLAG_TEST else N_DFS_TEST

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


def get_pd_col(freq: int) -> str:
    """
    Return column name for brightness temperature polarization difference.
    """
    return f"pd_{freq}"


def get_ratio_col(tc_num: str, tc_denom: str) -> str:
    """
    Return column name for the ratio of two brightness temperatures.
    """
    return f"ratio_{tc_num.removeprefix("Tc_")}_{tc_denom.removeprefix("Tc_")}"


def get_diff_col(tc_min: str, tc_sub: str) -> str:
    """
    Return column name for the difference between two brightness temperatures.
    """
    return f"diff_{tc_min.removeprefix('Tc_')}_{tc_sub.removeprefix('Tc_')}"


def _get_pd_expr(freq: int) -> pl.Expr:
    return pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")).alias(get_pd_col(freq))


def _get_ratio_expr(tc_num: str, tc_denom: str) -> pl.Expr:
    return pl.col(tc_num).truediv(pl.col(tc_denom)).alias(get_ratio_col(tc_num, tc_denom))


def _get_diff_expr(tc_min: str, tc_sub: str) -> pl.Expr:
    return pl.col(tc_min).sub(pl.col(tc_sub)).alias(get_diff_col(tc_min, tc_sub))


def _add_pd_unc(freq: int, unc_dict: Dict[str, float]):
    unc_dict[get_pd_col(freq)] = (unc_dict[f"Tc_{freq}V"] + unc_dict[f"Tc_{freq}H"]) / 2


def _add_ratio_unc(tc_num: str, tc_denom: str, unc_dict: Dict[str, float]):
    unc_dict[get_ratio_col(tc_num, tc_denom)] = (unc_dict[tc_num] + unc_dict[tc_denom]) / 100


def _add_diff_unc(tc_min: str, tc_sub: str, unc_dict: Dict[str, float]):
    unc_dict[get_diff_col(tc_min, tc_sub)] = (unc_dict[tc_min] + unc_dict[tc_sub]) / 2


def default_transform(obj, _: bool = True):
    return obj


def pd_transform(obj, drop: bool = True):
    """
    Replace vertical polarizations with polarization differences when possible.
    """
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_pd_expr(freq) for freq in [19, 37, 89, 165]])
        lf = lf.with_columns(_get_diff_expr("Tc_183V7", "Tc_183V3"))
        if drop:
            lf = lf.drop(TC_COLUMNS)
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        for freq in [19, 37, 89, 165]:
            _add_pd_unc(freq, unc_dict)
        _add_diff_unc("Tc_183V7", "Tc_183V3", unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_pd_col(freq) for freq in [19, 37, 89, 165]] + [get_diff_col("Tc_183V7", "Tc_183V3")]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def ratio_transform(obj, drop: bool = True):
    """
    Divide values by the values of 19H.
    """
    tc_denom = "Tc_19H"

    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr(tc_col, tc_denom) for tc_col in TC_COLUMNS if tc_col != tc_denom])
        if drop:
            lf = lf.drop(TC_COLUMNS)
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        for tc_col in TC_COLUMNS:
            if tc_col == tc_denom:
                continue
            _add_ratio_unc(tc_col, tc_denom, unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_ratio_col(tc_col, tc_denom) for tc_col in TC_COLUMNS if tc_col != tc_denom]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v1_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr("Tc_37H", "Tc_19H"), _get_pd_expr(89)])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V7"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        _add_ratio_unc("Tc_37H", "Tc_19H", unc_dict)
        _add_pd_unc(89, unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_ratio_col("Tc_37H", "Tc_19H"), get_pd_col(89), "Tc_23V", "Tc_165V", "Tc_183V7"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v2_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr("Tc_37H", "Tc_19H"), _get_pd_expr(89)])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_19V", "Tc_89V"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        _add_ratio_unc("Tc_37H", "Tc_19H", unc_dict)
        _add_pd_unc(89, unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_ratio_col("Tc_37H", "Tc_19H"), get_pd_col(89), "Tc_19V", "Tc_89V"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v3_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_pd_expr(165), _get_diff_expr("Tc_183V3", "Tc_183V7")])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V3"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        _add_pd_unc(165, unc_dict)
        _add_diff_unc("Tc_183V3", "Tc_183V7", unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_pd_col(165), get_diff_col("Tc_183V3", "Tc_183V7"), "Tc_23V", "Tc_165V", "Tc_183V3"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v4_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_pd_expr(19), _get_pd_expr(37),
                              _get_diff_expr("Tc_37V", "Tc_19V"),
                              _get_diff_expr("Tc_89V", "Tc_37V")])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_37V"]])
        return lf

    if isinstance(obj, Dict):
        unc_dict = obj
        _add_pd_unc(19, unc_dict)
        _add_pd_unc(37, unc_dict)
        _add_diff_unc("Tc_37V", "Tc_19V", unc_dict)
        _add_diff_unc("Tc_89V", "Tc_37V", unc_dict)
        return unc_dict

    if isinstance(obj, List):
        return [get_pd_col(19), get_pd_col(37),
                get_diff_col("Tc_37V", "Tc_19V"), get_diff_col("Tc_89V", "Tc_37V"),
                "Tc_37V"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def estimate_uncertainty_factor(path: pathlib.Path, transform: Callable, filter_rows: Callable, clip: bool):
    """
    Estimate a factor which uncertainty should be multiplied by during quantization.
    Write the estimated factor into a file in the parent directory of the specified path.
    """
    unc_dict = transform(get_uncertainties_dict(TC_COLUMNS))
    # TODO: rewrite to use transform
    range_dict = get_range_dict(clip)

    logging.info("Reading data")
    np.random.seed(566)

    x_bounds, y_bounds = _calculate_bounds(x_step=1, y_step=1)

    lfs = []
    k = 30
    for idx_x, idx_y in tqdm(zip(np.random.choice(np.arange(len(x_bounds) - 1), k),
                                 np.random.choice(np.arange(len(y_bounds) - 1), k)), total=k):
        extent = [x_bounds[idx_x], x_bounds[idx_x + 1], y_bounds[idx_y], y_bounds[idx_y + 1]]
        columns = TC_COLUMNS + [COLUMN_LON, COLUMN_LAT, COLUMN_TIME, VARIABLE_SURFACE_TYPE_INDEX]
        lf: pl.LazyFrame = gpm.bucket.read(DIR_BUCKET, extent, columns=columns, backend="polars")
        lfs.append(transform(filter_rows(lf)))

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

    n = len(lfs)
    while uncertainty_factor_r > uncertainty_factor_l + 1:
        factor = (uncertainty_factor_l + uncertainty_factor_r + 1) // 2
        logging.info("Checking uncertainty factor %d", factor)

        sizes_before = np.zeros(n)
        sizes_after = np.zeros(n)

        for i, lf in tqdm(enumerate(lfs)):
            curr_uns_dict = {k: factor * v for k, v in uncertainty_dict.items()}
            with disable_logging():
                lf_quantized = quantize_pmw_features(lf, quant_columns, curr_uns_dict, range_dict)

            # sizes_before[i] = lf.select(pl.len()).collect(engine="streaming").item()
            sizes_before[i] = lf.height
            # sizes_after[i] = lf_quantized.select(pl.len()).collect(engine="streaming").item()
            sizes_after[i] = lf_quantized.height

        if sizes_before.sum() == 0:
            raise ValueError("All data was filtered out before quantization.")

        mask_non_empty = sizes_before > 0
        sizes_before = sizes_before[mask_non_empty]
        sizes_after = sizes_after[mask_non_empty]
        before_to_after_ratio = np.median(sizes_before / sizes_after)
        logging.info("before/after: %.2f", before_to_after_ratio)
        if before_to_after_ratio < 40:
            uncertainty_factor_l = factor
        else:
            uncertainty_factor_r = factor
    return uncertainty_factor_r


def get_range_dict(clip: bool) -> Dict[str, float]:
    """
    Fetch or calculate a dictionary of ranges and return it.
    """
    if clip:
        ranges_dict = {}
        hists = get_feature_histograms()
        for col, hist in hists.items():
            value_count = hist.select(pl.col(COLUMN_BREAKPOINT).alias(col), pl.col(COLUMN_COUNT))
            value_count = value_count.filter(pl.col(col).is_not_null())
            value_count = value_count.sort(col)

            q_min, q_max = weighted_quantiles(value_count, [1e-6, 1 - 1e-6], col, COLUMN_COUNT)
            ranges_dict[f"{col}_min"] = q_min
            ranges_dict[f"{col}_max"] = q_max
        print(ranges_dict)
        return
    tc_denom = "Tc_19H"
    diff_columns = ["Tc_23V", "Tc_183V7", "Tc_183V3", "Tc_165V", "Tc_89V", "Tc_37V", "Tc_19V", "Tc_10V"]
    columns = (
            TC_COLUMNS +
            [get_pd_col(freq) for freq in [10, 19, 37, 89, 165]] +
            [get_ratio_col(tc_num, tc_denom) for tc_num in TC_COLUMNS if tc_num != tc_denom] +
            [get_diff_col(tc_min, tc_sub) for tc_min, tc_sub in zip(diff_columns, diff_columns[1:])]
    )
    range_dict = (
            {f"{col}_min": -np.inf for col in columns} |
            {f"{col}_max": np.inf for col in columns}
    )
    return range_dict


def get_feature_histograms() -> Dict[str, pl.DataFrame]:
    hists_path = pathlib.Path(DIR_PMW_ANALYSIS) / "feature_histograms.pkl"
    if hists_path.exists():
        hists = pickle.load(open(hists_path, "rb"))
        return hists

    logging.info("Calculating feature histograms.")
    hists = _calculate_feature_histograms()
    pickle.dump(hists, open(hists_path, 'wb'))
    return hists


_FILE_SUFFIX_HIST_PARQUET = "_hist.parquet"


def _calculate_feature_histograms() -> Dict[str, pl.DataFrame]:
    multiprocessing.set_start_method("spawn", force=True)

    x_bounds, y_bounds = _calculate_bounds()

    tc_denom = "Tc_19H"
    diff_columns = ["Tc_23V", "Tc_183V7", "Tc_183V3", "Tc_165V", "Tc_89V", "Tc_37V", "Tc_19V", "Tc_10V"]

    pd_freqs = [10, 19, 37, 89, 165]
    expressions = (
            [_get_pd_expr(freq) for freq in pd_freqs] +
            [_get_ratio_expr(tc_num, tc_denom) for tc_num in TC_COLUMNS if tc_num != tc_denom] +
            [_get_diff_expr(tc_min, tc_sub) for tc_min, tc_sub in zip(diff_columns, diff_columns[1:])]
    )

    unc_dict = get_uncertainties_dict(TC_COLUMNS)
    # update uncertainties dictionary to have pd, ratio and diff
    for pd_freq in pd_freqs:
        _add_pd_unc(pd_freq, unc_dict)
    for tc_num in TC_COLUMNS:
        if tc_num != tc_denom:
            _add_ratio_unc(tc_num, tc_denom, unc_dict)
    for tc_min, tc_sub in zip(diff_columns, diff_columns[1:]):
        _add_diff_unc(tc_min, tc_sub, unc_dict)

    col_to_hists = defaultdict(lambda: [])

    path = pathlib.Path(DIR_PMW_ANALYSIS) / "hists_tmp"
    if path.exists():
        rmtree(path)
    path.mkdir(parents=True)

    progress_bar = tqdm(total=(len(x_bounds) - 1) * (len(y_bounds) - 1))
    for idx_x, (x_min, x_max) in enumerate(zip(x_bounds[:-1], x_bounds[1:])):
        for idx_y, (y_min, y_max) in enumerate(zip(y_bounds[:-1], y_bounds[1:])):
            idx = idx_x * (len(y_bounds) - 1) + idx_y

            if FLAG_TEST and idx >= N_DFS_TEST:
                break

            extent = [x_min, x_max, y_min, y_max]

            p = multiprocessing.Process(target=_calculate_feature_histograms_subprocess,
                                        args=(expressions, unc_dict, path, extent))
            p.start()
            p.join()
            p.close()

            for df_path in path.iterdir():
                name = df_path.name

                if name.endswith(_FILE_SUFFIX_HIST_PARQUET):
                    col = name.removesuffix(_FILE_SUFFIX_HIST_PARQUET)
                    df_hist = pl.read_parquet(df_path)
                    col_to_hists[col].append(df_hist)

            progress_bar.update(1)

        logging.info(get_memory_usage())
    rmtree(path)

    hist_dict = {}
    pl_hist_columns = [COLUMN_BREAKPOINT, COLUMN_CATEGORY]
    for col, dfs_list in col_to_hists.items():
        hist_dict[f"{col}"] = pl.concat(dfs_list).group_by(pl_hist_columns).agg(pl.col(COLUMN_COUNT).sum())

    return hist_dict


def _calculate_feature_histograms_subprocess(expressions, unc_dict, path, extent):
    df: pl.DataFrame = gpm.bucket.read(DIR_BUCKET, extent=extent, backend="polars",
                                       columns=TC_COLUMNS + [COLUMN_LON, COLUMN_LAT])
    df = df.drop([COLUMN_LON, COLUMN_LAT])
    df = df.with_columns(expressions)

    for col in df.columns:
        unc = unc_dict[col]
        v_min = df[col].min()
        v_max = df[col].max()
        bins = np.arange(int(np.floor(v_min / unc)), int(np.ceil(v_max / unc)) + 1) * unc
        df_hist = df[col].hist(bins=bins)
        df_hist.write_parquet(path / f"{col}{_FILE_SUFFIX_HIST_PARQUET}")

    logging.info(get_memory_usage())


def get_newest_k(path: pathlib.Path, transform: Callable, k: int):
    df_id = pl.read_parquet(path / FILE_DF_FINAL)

    # 1. Get observations before quantization
    df_id_k = take_k_sorted(df_id, COLUMN_OCCURRENCE, k, COLUMN_COUNT, descending=True)
    df_k = _get_bucket_data_for_ids(df_id_k, transform)

    # 2. Add a column to mark sun glint presence
    df_k = create_occurrence_column(df_k)
    df_k, sun_glint_column = retrieve_possible_sun_glint(df_k)

    df_k.write_parquet(path / FILE_DF_FINAL_K)

    # 3. Store observations excluding the ones affected by sun glint
    dir_no_sun_glint = path / DIR_NO_SUN_GLINT
    dir_no_sun_glint.mkdir(parents=True, exist_ok=True)
    df_k_no_sun_glint = df_k.filter(~pl.col(sun_glint_column)).drop(sun_glint_column)

    df_k_no_sun_glint.write_parquet(dir_no_sun_glint / "final_k.parquet")


def _get_bucket_data_for_ids(df_id_k: pl.DataFrame, transform: Callable) -> pl.DataFrame:
    id_columns = [COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID]
    quant_columns = transform(TC_COLUMNS)

    p: LonLatPartitioning = get_bucket_spatial_partitioning(DIR_BUCKET)

    df_id_k = df_id_k.select([COLUMN_LON, COLUMN_LAT] + id_columns + quant_columns)
    df_id_k = df_id_k.explode([COLUMN_LON, COLUMN_LAT] + id_columns)
    df_id_k = df_id_k.rename({col: f"{col}{COLUMN_SUFFIX_QUANT}" for col in quant_columns})

    df_id_k = p.add_labels(df_id_k, x=COLUMN_LON, y=COLUMN_LAT)
    df_id_k = p.add_centroids(df_id_k, x=COLUMN_LON, y=COLUMN_LAT, x_coord=COLUMN_LON_BIN, y_coord=COLUMN_LAT_BIN)

    df_id_k_grouped = df_id_k.group_by(p.levels)
    agg_off_columns = [col for col in df_id_k.columns if col not in p.levels]
    df_id_k_agg = df_id_k_grouped.agg(pl.col(agg_off_columns)).sort(p.levels)

    dfs_k_bin = []

    progress_bar = tqdm(total=(len(p.x_bounds) - 1) * (len(p.y_bounds) - 1))
    for x_min, x_max, x_c in zip(p.x_bounds[:-1], p.x_bounds[1:], p.x_centroids):
        for y_min, y_max, y_c in zip(p.y_bounds[:-1], p.y_bounds[1:], p.y_centroids):
            if FLAG_TEST and len(dfs_k_bin) >= N_DFS_TEST:
                break

            df_k_bin = df_id_k_agg.filter(pl.col(COLUMN_LON_BIN) == x_c, pl.col(COLUMN_LAT_BIN) == y_c).drop(p.levels)
            df_k_bin = df_k_bin.explode(agg_off_columns)

            if df_k_bin.is_empty():
                progress_bar.update(1)
                continue

            extent = [x_min, x_max, y_min, y_max]
            df_bin = gpm.bucket.read(bucket_dir=DIR_BUCKET,
                                     extent=extent,
                                     backend="polars")
            df_k_bin = df_k_bin.join(df_bin, on=id_columns, how="inner")
            dfs_k_bin.append(df_k_bin)
            progress_bar.update(1)

    if len(dfs_k_bin) == 0:
        schema = (
                {col: pl.Float32 for col in quant_columns} |
                {COLUMN_LON: pl.Float32, COLUMN_LAT: pl.Float32, COLUMN_TIME: pl.Datetime} |
                {VARIABLE_SURFACE_TYPE_INDEX: pl.UInt32, COLUMN_L1C_QUALITY_FLAG: pl.Int32}
        )
        return pl.DataFrame(schema=schema)

    df_k = pl.concat(dfs_k_bin)
    df_k = transform(df_k, drop=False)
    return df_k


def get_transformation_function(arg_transform: ArgTransform) -> Callable:
    """
    Return a transformation function based on the specified argument.
    """
    match arg_transform:
        case ArgTransform.DEFAULT:
            transform = default_transform
        case ArgTransform.PD:
            transform = pd_transform
        case ArgTransform.RATIO:
            transform = ratio_transform
        case ArgTransform.V1:
            transform = v1_transform
        case ArgTransform.V2:
            transform = v2_transform
        case ArgTransform.V3:
            transform = v3_transform
        case ArgTransform.V4:
            transform = v4_transform
        case _:
            raise ValueError(f"{arg_transform.value} is not supported.")
    return transform


def main():
    logging.basicConfig(level=logging.INFO)

    parser = configargparse.ArgumentParser(config_arg_is_required=True, args_for_setting_config_path=["--config"],
                                           description="Run quantization")

    parser.add_argument("--step", type=ArgQuantizationStep, action=EnumAction,
                        help="Quantization pipeline's step to perform")
    parser.add_argument("--transform", default=ArgTransform.DEFAULT, type=ArgTransform, action=EnumAction,
                        help="Type of transformation to perform on data")
    parser.add_argument("--dir", default=DIR_PMW_ANALYSIS,
                        help="Path to the directory to store quantized data in")
    parser.add_argument("--agg-off-cols", default=[], nargs="+",
                        help="Columns whose values are stored in lists, without aggregation")
    parser.add_argument("--month", type=int, help="Month of the data to quantize")
    parser.add_argument("--year", type=int, help="Year of the data to quantize")
    parser.add_argument("--l2-l3-columns", default=ArgQuantizationL2L3Columns.ALL,
                        type=ArgQuantizationL2L3Columns, action=EnumAction,
                        help="L2 and L3 columns to process during quantization")
    parser.add_argument("--surface-type", default=ArgSurfaceType.ALL,
                        type=ArgSurfaceType, action=EnumAction,
                        help="Surface type to process during quantization")
    parser.add_argument("--clip", type=bool, default=False,
                        help="If true, data is clipped to the range from range dictionary")

    args = parser.parse_args()

    assert not (args.year is None and args.month is not None)

    path = pathlib.Path(args.dir) / args.transform.value / args.surface_type.value
    path.mkdir(parents=True, exist_ok=True)

    transform = get_transformation_function(args.transform)
    filter_rows = lambda df: filter_by_surface_type(df, args.surface_type.indexes())

    if args.step == ArgQuantizationStep.FACTOR:
        estimate_uncertainty_factor(path, transform, filter_rows, args.clip)
        return

    with open(path / "factor", "r", encoding="utf-8") as file:
        factor = float(file.read())

    if args.year is not None:
        path = path / str(args.year)
    if args.month is not None:
        path = path / str(args.month)
    path.mkdir(parents=True, exist_ok=True)

    match args.step:
        case ArgQuantizationStep.QUANTIZE:
            quantize(path, transform, filter_rows, factor, args.clip,
                     args.month, args.year, args.agg_off_cols, args.l2_l3_columns)
        case ArgQuantizationStep.MERGE:
            merge(path, transform, args.agg_off_cols)
        case ArgQuantizationStep.NEWEST_K:
            get_newest_k(path, transform, K)
        case _:
            raise ValueError(f"{args.step.value} is not supported.")


if __name__ == '__main__':
    main()
