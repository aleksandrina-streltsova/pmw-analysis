"""Shared transformation helpers for quantization scripts."""

from typing import Callable

import polars as pl

from pmw_analysis.constants import ArgTransform, TC_COLUMNS


def get_pd_col(freq: int) -> str:
    """Return column name for brightness temperature polarization difference."""
    return f"pd_{freq}"


def get_ratio_col(tc_num: str, tc_denom: str) -> str:
    """Return column name for the ratio of two brightness temperatures."""
    return f"ratio_{tc_num.removeprefix('Tc_')}_{tc_denom.removeprefix('Tc_')}"


def get_diff_col(tc_min: str, tc_sub: str) -> str:
    """Return column name for the difference between two brightness temperatures."""
    return f"diff_{tc_min.removeprefix('Tc_')}_{tc_sub.removeprefix('Tc_')}"


def _get_pd_expr(freq: int) -> pl.Expr:
    return pl.col(f"Tc_{freq}V").sub(pl.col(f"Tc_{freq}H")).alias(get_pd_col(freq))


def _get_ratio_expr(tc_num: str, tc_denom: str) -> pl.Expr:
    return pl.col(tc_num).truediv(pl.col(tc_denom)).alias(get_ratio_col(tc_num, tc_denom))


def _get_diff_expr(tc_min: str, tc_sub: str) -> pl.Expr:
    return pl.col(tc_min).sub(pl.col(tc_sub)).alias(get_diff_col(tc_min, tc_sub))


def _add_pd_unc(freq: int, unc_dict: dict[str, float]):
    unc_dict[get_pd_col(freq)] = (unc_dict[f"Tc_{freq}V"] + unc_dict[f"Tc_{freq}H"]) / 2


def _add_ratio_unc(tc_num: str, tc_denom: str, unc_dict: dict[str, float]):
    unc_dict[get_ratio_col(tc_num, tc_denom)] = (unc_dict[tc_num] + unc_dict[tc_denom]) / 100


def _add_diff_unc(tc_min: str, tc_sub: str, unc_dict: dict[str, float]):
    unc_dict[get_diff_col(tc_min, tc_sub)] = (unc_dict[tc_min] + unc_dict[tc_sub]) / 2


def default_transform(obj, _: bool = True):
    return obj


def pd_transform(obj, drop: bool = True):
    """Replace vertical polarizations with polarization differences when possible."""
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_pd_expr(freq) for freq in [19, 37, 89, 165]])
        lf = lf.with_columns(_get_diff_expr("Tc_183V7", "Tc_183V3"))
        if drop:
            lf = lf.drop(TC_COLUMNS)
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        for freq in [19, 37, 89, 165]:
            _add_pd_unc(freq, unc_dict)
        _add_diff_unc("Tc_183V7", "Tc_183V3", unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [get_pd_col(freq) for freq in [19, 37, 89, 165]] + [get_diff_col("Tc_183V7", "Tc_183V3")]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def ratio_transform(obj, drop: bool = True):
    """Divide values by the values of 19H."""
    tc_denom = "Tc_19H"

    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr(tc_col, tc_denom) for tc_col in TC_COLUMNS if tc_col != tc_denom])
        if drop:
            lf = lf.drop(TC_COLUMNS)
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        for tc_col in TC_COLUMNS:
            if tc_col == tc_denom:
                continue
            _add_ratio_unc(tc_col, tc_denom, unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [get_ratio_col(tc_col, tc_denom) for tc_col in TC_COLUMNS if tc_col != tc_denom]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v1_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr("Tc_37H", "Tc_19H"), _get_pd_expr(89)])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V7"]])
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        _add_ratio_unc("Tc_37H", "Tc_19H", unc_dict)
        _add_pd_unc(89, unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [get_ratio_col("Tc_37H", "Tc_19H"), get_pd_col(89), "Tc_23V", "Tc_165V", "Tc_183V7"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v2_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_ratio_expr("Tc_37H", "Tc_19H"), _get_pd_expr(89)])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_19V", "Tc_89V"]])
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        _add_ratio_unc("Tc_37H", "Tc_19H", unc_dict)
        _add_pd_unc(89, unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [get_ratio_col("Tc_37H", "Tc_19H"), get_pd_col(89), "Tc_19V", "Tc_89V"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v3_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([_get_pd_expr(165), _get_diff_expr("Tc_183V3", "Tc_183V7")])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_23V", "Tc_165V", "Tc_183V3"]])
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        _add_pd_unc(165, unc_dict)
        _add_diff_unc("Tc_183V3", "Tc_183V7", unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [get_pd_col(165), get_diff_col("Tc_183V3", "Tc_183V7"), "Tc_23V", "Tc_165V", "Tc_183V3"]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def v4_transform(obj, drop: bool = True):
    if isinstance(obj, (pl.DataFrame, pl.LazyFrame)):
        lf = obj
        lf = lf.with_columns([
            _get_pd_expr(19),
            _get_pd_expr(37),
            _get_diff_expr("Tc_37V", "Tc_19V"),
            _get_diff_expr("Tc_89V", "Tc_37V"),
        ])
        if drop:
            lf = lf.drop([col for col in TC_COLUMNS if col not in ["Tc_37V"]])
        return lf

    if isinstance(obj, dict):
        unc_dict = obj
        _add_pd_unc(19, unc_dict)
        _add_pd_unc(37, unc_dict)
        _add_diff_unc("Tc_37V", "Tc_19V", unc_dict)
        _add_diff_unc("Tc_89V", "Tc_37V", unc_dict)
        return unc_dict

    if isinstance(obj, list):
        return [
            get_pd_col(19),
            get_pd_col(37),
            get_diff_col("Tc_37V", "Tc_19V"),
            get_diff_col("Tc_89V", "Tc_37V"),
            "Tc_37V",
        ]

    raise TypeError("Unsupported object type: " + str(type(obj)) + ". Supported types: pl.DataFrame, Dict.")


def get_transformation_function(arg_transform: ArgTransform) -> Callable:
    """Return a transformation function based on the specified argument."""
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
