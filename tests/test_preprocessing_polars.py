"""
This module contains tests for data preprocessing functionalities.
"""
import datetime
import unittest
from typing import Dict, List, Sequence

import numpy as np
import polars as pl

from pmw_analysis.constants import COLUMN_COUNT, STRUCT_FIELD_COUNT, COLUMN_OCCURRENCE, TC_COLUMNS, Stats
from pmw_analysis.quantization.dataframe_polars import _round, _get_tc_columns, \
    _get_special_dict, _aggregate, _get_periodic_dict, merge_quantized_pmw_features, \
    DataFrameQuantizationInfo, get_agg_column


def _get_test_columns() -> Sequence[str]:
    return [
        'L1CqualityFlag', 'Quality_HF', 'Quality_LF', 'SClatitude', 'SCorientation', 'Tc_10H', 'Tc_10V', 'Tc_165H',
        'Tc_165V', 'Tc_183V3', 'Tc_183V7', 'Tc_19H', 'Tc_19V', 'Tc_23V', 'Tc_37H', 'Tc_37V', 'Tc_89H', 'Tc_89V',
        'airmassLiftIndex', 'cloudWaterPath', 'convectivePrecipitation', 'day', 'frozenPrecipitation', 'gpm_id',
        'iceWaterPath', 'incidenceAngle_HF', 'incidenceAngle_LF', 'lat', 'lon', 'mostLikelyPrecipitation',
        'pixelStatus', 'precip1stTertial', 'precip2ndTertial', 'precipitationYesNoFlag', 'probabilityOfPrecip',
        'qualityFlag', 'rainWaterPath', 'sunGlintAngle_HF', 'sunGlintAngle_LF', 'sunLocalTime', 'surfacePrecipitation',
        'surfaceTypeIndex', 'temp2mIndex', 'time', 'totalColumnWaterVaporIndex'
    ]


def _get_agg_off_columns() -> Sequence[str]:
    return ['gpm_id', 'lon', 'time']


def _create_single_row() -> pl.DataFrame:
    columns = _get_test_columns()
    df = pl.DataFrame({**{col: 0. for col in columns},
                       COLUMN_OCCURRENCE: datetime.datetime(2000, 1, 1, 0, 0, 0)})
    return df


def _create_single_row_aggregated() -> pl.DataFrame:
    columns = _get_test_columns()
    tc_columns = _get_tc_columns(columns)
    agg_off_columns = _get_agg_off_columns()

    info = DataFrameQuantizationInfo.create(columns, tc_columns, agg_off_columns=agg_off_columns)

    special_dict = _get_special_dict()

    tc = {col: 0. for col in tc_columns}
    flag = {col: [[]] for col in info.flag_columns}

    periodic = {k: v
                for col in info.periodic_columns
                for k, v in {f"{col}_lt": 0., f"{col}_lt_count": 0, f"{col}_gt": 0., f"{col}_gt_count": 0}.items()}

    special = {k: v
               for col in info.special_columns
               for k, v in {col: 0., f"{col}_{special_dict[col][1]}_count": 0, f"{col}_count": 0}.items()}

    agg_off = {col: [[0.]] for col in info.agg_off_columns}

    agg_mean = {k: v
                for col in info.get_agg_mean_columns(columns)
                for k, v in {col: 0., f"{col}_count": 0.}.items()}

    df = pl.DataFrame(data={
        **tc,
        **flag,
        **periodic,
        **special,
        **special,
        **agg_off,
        **agg_mean,
        get_agg_column(COLUMN_OCCURRENCE, Stats.MIN): datetime.datetime(2000, 1, 1, 0, 0, 0),
        get_agg_column(COLUMN_OCCURRENCE, Stats.MAX): datetime.datetime(2001, 1, 1, 1, 1, 1),
        COLUMN_COUNT: 0,
    }).with_columns([
        pl.col(flag_col)
        .cast(pl.List(pl.Struct([
            pl.Field(flag_col, pl.Int16),
            pl.Field(STRUCT_FIELD_COUNT, pl.Int64)
        ]))) for flag_col in info.flag_columns
    ])
    return df


def _append_struct(df: pl.DataFrame, structs: List[Dict], column: str, row: int):
    df = df.with_columns(
        pl.when(pl.arange(0, pl.len()) == row)
        .then(pl.col(column).list.concat(pl.lit(structs, dtype=pl.List(pl.Struct))))
        .otherwise(pl.col(column))
    )
    return df


def _assign_value(df, value, col, row):
    return df.with_columns(
        pl.when(pl.arange(0, pl.len()) == row)
        .then(value)
        .otherwise(pl.col(col))
        .alias(col)
    )


class PreprocessingPolarsTestCase(unittest.TestCase):
    """
    Unit test case class for testing data preprocessing functionalities.
    """

    def test_round(self) -> None:
        """
        Unit test for verifying the correctness of the `pmw_analysis.quantization.dataframe_polars._round` function.
        """
        tc_columns = _get_tc_columns(_get_test_columns())

        df_before = _create_single_row()

        n_tc = len(tc_columns)
        df_before = pl.concat([df_before for _ in range(3)])

        for i in range(n_tc):
            df_before[0, tc_columns[i]] = 0.3 + i
            df_before[1, tc_columns[i]] = 0.3 + 2 * i
            df_before[2, tc_columns[i]] = 0.3 + 2 * i
        df_before[2, tc_columns[n_tc - 1]] = (0.5 + (n_tc - 1) * 1.5) * 2 + 1

        uncertainties = [0.5 + i * 1.5 for i, col in enumerate(tc_columns)]

        df_after_expected = df_before.clone()
        for i in range(n_tc):
            df_after_expected[0, tc_columns[i]] = 0.5 + i * 1.5
            df_after_expected[1, tc_columns[i]] = 0.5 + i * 1.5
            df_after_expected[2, tc_columns[i]] = 0.5 + i * 1.5
        df_after_expected[2, tc_columns[n_tc - 1]] = (0.5 + (n_tc - 1) * 1.5) * 2

        # lf_before = df_before.lazy()
        lf_before = df_before
        lf_after_actual = _round(lf_before, tc_columns, uncertainties, quant_ranges=None)

        df_after_actual = lf_after_actual  # .collect()

        assert df_after_expected.equals(df_after_actual)

    def test_aggregate(self):
        """
        Unit test for verifying the correctness of the `pmw_analysis.quantization.dataframe_polars._aggregate` function.
        """
        columns = _get_test_columns()
        tc_cols = _get_tc_columns(columns)
        agg_off_columns = _get_agg_off_columns()

        info = DataFrameQuantizationInfo.create(columns, tc_cols,
                                                agg_off_columns=agg_off_columns,
                                                periodic_dict=_get_periodic_dict(),
                                                special_dict=_get_special_dict(),
                                                agg_off_limit=4)

        agg_mean_columns = info.get_agg_mean_columns(columns)

        #### df before aggregation ####
        n = 6
        df_before = pl.concat([_create_single_row() for _ in range(n)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(n):
                df_before[j, tc_col] = i
        df_before[n - 1, tc_cols[0]] = 1
        # flag
        for i, flag_col in enumerate(info.flag_columns):
            df_before[0, flag_col] = i
            df_before[1, flag_col] = i
            df_before[2, flag_col] = i
            df_before[3, flag_col] = i + 1
            df_before[4, flag_col] = None
        df_before[0, info.flag_columns[0]] = -1
        # periodic
        for periodic_col in info.periodic_columns:
            mid = info.periodic_dict[periodic_col]
            df_before[0, periodic_col] = mid - 1
            df_before[1, periodic_col] = mid - 1
            df_before[2, periodic_col] = mid
            df_before[3, periodic_col] = mid + 1
            df_before[4, periodic_col] = None
        # special
        for i, special_col in enumerate(info.special_columns):
            df_before[0, special_col] = i
            df_before[1, special_col] = i + 1
            df_before[2, special_col] = info.special_dict[special_col][0]
            df_before[3, special_col] = info.special_dict[special_col][0]
            df_before[4, special_col] = None
        # agg_off
        for i, agg_off_col in enumerate(info.agg_off_columns):
            df_before[0, agg_off_col] = i
            df_before[1, agg_off_col] = i
            df_before[2, agg_off_col] = i
            df_before[3, agg_off_col] = i + 1
            df_before[4, agg_off_col] = None
        df_before[0, info.agg_off_columns[0]] = -1
        # agg_mean
        for i, agg_mean_col in enumerate(agg_mean_columns):
            df_before[0, agg_mean_col] = i
            df_before[1, agg_mean_col] = i
            df_before[2, agg_mean_col] = i + 1
            df_before[3, agg_mean_col] = None
            df_before[4, agg_mean_col] = None
        # time
        df_before[0, COLUMN_OCCURRENCE] = datetime.datetime(2020, 1, 1, 0, 0, 0)
        df_before[1, COLUMN_OCCURRENCE] = datetime.datetime(2019, 2, 1, 0, 0, 0)
        df_before[2, COLUMN_OCCURRENCE] = datetime.datetime(2019, 1, 2, 0, 0, 0)
        df_before[3, COLUMN_OCCURRENCE] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        df_before[4, COLUMN_OCCURRENCE] = datetime.datetime(2019, 1, 1, 1, 0, 0)
        df_before[5, COLUMN_OCCURRENCE] = datetime.datetime(2019, 1, 1, 1, 0, 0)

        #### expected df after aggregation ####
        expected = pl.concat([_create_single_row_aggregated() for _ in range(2)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(2):
                expected[j, tc_col] = i
        expected[1, tc_cols[0]] = 1
        # flag
        for i, flag_col in enumerate(info.flag_columns):
            if i == 0:
                expected = _append_struct(expected,
                                          [{flag_col: None, STRUCT_FIELD_COUNT: 1},
                                           {flag_col: -1, STRUCT_FIELD_COUNT: 1},
                                           {flag_col: i, STRUCT_FIELD_COUNT: 2},
                                           {flag_col: i + 1, STRUCT_FIELD_COUNT: 1}],
                                          flag_col, row=0)
            else:
                expected = _append_struct(expected,
                                          [{flag_col: None, STRUCT_FIELD_COUNT: 1},
                                           {flag_col: i, STRUCT_FIELD_COUNT: 3},
                                           {flag_col: i + 1, STRUCT_FIELD_COUNT: 1}], flag_col, 0)
            expected = _append_struct(expected, [{flag_col: 0, STRUCT_FIELD_COUNT: 1}], flag_col, 1)
        # periodic
        for i, periodic_col in enumerate(info.periodic_columns):
            expected[0, f"{periodic_col}_lt"] = info.periodic_dict[periodic_col] - 2 / 3
            expected[0, f"{periodic_col}_lt_count"] = 3
            expected[0, f"{periodic_col}_gt"] = info.periodic_dict[periodic_col] + 1
            expected[0, f"{periodic_col}_gt_count"] = 1

            expected[1, f"{periodic_col}_lt"] = 0
            expected[1, f"{periodic_col}_lt_count"] = 1
            expected[1, f"{periodic_col}_gt"] = None
            expected[1, f"{periodic_col}_gt_count"] = 0
        # special
        for i, special_col in enumerate(info.special_columns):
            expected[0, special_col] = i + 0.5
            expected[0, f"{special_col}_{info.special_dict[special_col][1]}_count"] = 2
            expected[0, f"{special_col}_count"] = 2

            expected[1, f"{special_col}_count"] = 1
        # agg_off
        for i, agg_off_col in enumerate(info.agg_off_columns):
            if i == 0:
                row0 = [-1, i, i, i + 1]
            else:
                row0 = [i, i, i, i + 1]
            expected = _assign_value(expected, pl.lit(row0), agg_off_col, row=0)
            expected = _assign_value(expected, pl.lit([0]), agg_off_col, row=1)
        # other
        for i, agg_mean_col in enumerate(agg_mean_columns):
            expected[0, agg_mean_col] = i + 1 / 3
            expected[0, f"{agg_mean_col}_count"] = 3

            expected[1, f"{agg_mean_col}_count"] = 1
        # time
        expected[0, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        expected[0, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2020, 1, 1, 0, 0, 0)
        expected[1, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 1, 0, 0)
        expected[1, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2019, 1, 1, 1, 0, 0)
        # count
        expected[0, COLUMN_COUNT] = n - 1
        expected[1, COLUMN_COUNT] = 1

        # lf_before = df_before.lazy()
        lf_before = df_before
        actual = _aggregate(lf_before, info)  # .collect()
        actual = actual.with_columns([
            pl.col(flag_col).list.sort()
            for flag_col in info.flag_columns
        ])
        actual = actual.sort(by=COLUMN_COUNT, descending=True)

        # TODO should the order be preserved?
        self.assertEqual(sorted(expected.columns), sorted(actual.columns))
        # TODO replace with self.assert... ?
        assert expected.equals(actual.select(expected.columns))

    def test_merge(self):
        """
        Unit test for verifying the correctness of the
        `pmw_analysis.quantization.dataframe_polars.merge_quantized_pmw_features` function.
        """
        columns = _get_test_columns()
        tc_cols = _get_tc_columns(columns)
        agg_off_columns = _get_agg_off_columns()

        info = DataFrameQuantizationInfo.create(columns, tc_cols,
                                                agg_off_columns=agg_off_columns,
                                                periodic_dict=_get_periodic_dict(),
                                                special_dict=_get_special_dict())

        agg_mean_columns = info.get_agg_mean_columns(columns)

        #### dfs before merging ####
        n = 6
        dfs = [pl.concat([_create_single_row_aggregated() for _ in range(2)]) for _ in range(2)]

        # test that order of columns doesn't affect merge
        dfs[0] = dfs[0].select(dfs[0].columns[::-1])

        for k in range(len(dfs)):
            # tc
            for i, tc_col in enumerate(tc_cols):
                for j in range(2):
                    dfs[k][j, tc_col] = i
                dfs[k][1, tc_cols[0]] = k + 1
            # flag
            for i, flag_col in enumerate(info.flag_columns):
                if i == 0:
                    dfs[k] = _append_struct(
                        dfs[k],
                        structs=[
                            {flag_col: None, STRUCT_FIELD_COUNT: 1 * (k + 1)},
                            {flag_col: -1 * (k + 1), STRUCT_FIELD_COUNT: 1 * (k + 1)},
                            {flag_col: i, STRUCT_FIELD_COUNT: 2 * (k + 1)},
                            {flag_col: i + 1, STRUCT_FIELD_COUNT: 1 * (k + 1)}
                        ],
                        column=flag_col,
                        row=0,
                    )
                else:
                    dfs[k] = _append_struct(
                        dfs[k],
                        structs=[
                            {flag_col: None, STRUCT_FIELD_COUNT: 1 * (k + 1)},
                            {flag_col: i, STRUCT_FIELD_COUNT: 3 * (k + 1)},
                            {flag_col: i + 1, STRUCT_FIELD_COUNT: 1 * (k + 1)}
                        ],
                        column=flag_col,
                        row=0,
                    )
                dfs[k] = _append_struct(dfs[k], [{flag_col: 0, STRUCT_FIELD_COUNT: 1 * (k + 1)}], flag_col, 1)
            # periodic
            for i, periodic_col in enumerate(info.periodic_columns):
                dfs[k][0, f"{periodic_col}_lt"] = info.periodic_dict[periodic_col] - 1.5 - 3 * k
                dfs[k][0, f"{periodic_col}_lt_count"] = 3 * (k + 1)
                dfs[k][0, f"{periodic_col}_gt"] = info.periodic_dict[periodic_col] + 1.5 + 3 * k
                dfs[k][0, f"{periodic_col}_gt_count"] = 1 * (k + 1)

                dfs[k][1, f"{periodic_col}_lt"] = 0
                dfs[k][1, f"{periodic_col}_lt_count"] = 1 * (k + 1)
                dfs[k][1, f"{periodic_col}_gt"] = None
                dfs[k][1, f"{periodic_col}_gt_count"] = 0
            # special
            for i, special_col in enumerate(info.special_columns):
                dfs[k][0, special_col] = i + 0.5 + 3 * k
                dfs[k][0, f"{special_col}_{info.special_dict[special_col][1]}_count"] = 2 * (k + 1)
                dfs[k][0, f"{special_col}_count"] = 2 * (k + 1)

                dfs[k][1, f"{special_col}_count"] = 1 * (k + 1)
            # agg_off
            for i, agg_off_col in enumerate(info.agg_off_columns):
                if i == 0:
                    row0 = [-1, i, i, i + 1, None] * (k + 1)
                else:
                    row0 = [i, i, i, i + 1, None] * (k + 1)
                dfs[k] = _assign_value(dfs[k], pl.lit(row0), agg_off_col, row=0)
                dfs[k] = _assign_value(dfs[k], pl.lit([0] * (k + 1)), agg_off_col, row=1)
            # agg_mean
            for i, agg_mean_col in enumerate(agg_mean_columns):
                dfs[k][0, agg_mean_col] = i + 0.25 + 3 * k
                dfs[k][0, f"{agg_mean_col}_count"] = 3 * (k + 1)

                dfs[k][1, f"{agg_mean_col}_count"] = 1 * (k + 1)
            # count
            dfs[k][0, COLUMN_COUNT] = (n - 1) * (k + 1)
            dfs[k][1, COLUMN_COUNT] = k + 1
            # time
            dfs[k][0, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 1 - k, 0, 0)
            dfs[k][0, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2019, 1, 1, 1 - k, 1, 1)
            dfs[k][1, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 1, 0, 0)
            dfs[k][1, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2020, 1, 1, 1, 0, 0)

        #### expected df after merging ####
        expected = pl.concat([_create_single_row_aggregated() for _ in range(3)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(3):
                expected[j, tc_col] = i
            expected[1, tc_cols[0]] = 1
            expected[2, tc_cols[0]] = 2
        # flag
        for i, flag_col in enumerate(info.flag_columns):
            if i == 0:
                expected = _append_struct(
                    expected,
                    structs=[
                        {flag_col: None, STRUCT_FIELD_COUNT: 1 * 3},
                        {flag_col: -1 * 2, STRUCT_FIELD_COUNT: 1 * 2},
                        {flag_col: -1, STRUCT_FIELD_COUNT: 1 * 1},
                        {flag_col: i, STRUCT_FIELD_COUNT: 2 * 3},
                        {flag_col: i + 1, STRUCT_FIELD_COUNT: 1 * 3}
                    ],
                    column=flag_col,
                    row=0,
                )
            else:
                expected = _append_struct(
                    expected,
                    structs=[
                        {flag_col: None, STRUCT_FIELD_COUNT: 1 * 3},
                        {flag_col: i, STRUCT_FIELD_COUNT: 3 * 3},
                        {flag_col: i + 1, STRUCT_FIELD_COUNT: 1 * 3}
                    ],
                    column=flag_col,
                    row=0,
                )
            expected = _append_struct(expected, [{flag_col: 0, STRUCT_FIELD_COUNT: 1 * 1}], flag_col, 1)
            expected = _append_struct(expected, [{flag_col: 0, STRUCT_FIELD_COUNT: 1 * 2}], flag_col, 2)
        # periodic
        for i, periodic_col in enumerate(info.periodic_columns):
            expected[0, f"{periodic_col}_lt"] = info.periodic_dict[periodic_col] - 1.5 - 2
            expected[0, f"{periodic_col}_lt_count"] = 3 * 3
            expected[0, f"{periodic_col}_gt"] = info.periodic_dict[periodic_col] + 1.5 + 2
            expected[0, f"{periodic_col}_gt_count"] = 1 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{periodic_col}_lt"] = 0
                expected[k + 1, f"{periodic_col}_lt_count"] = 1 * (k + 1)
                expected[k + 1, f"{periodic_col}_gt"] = np.NaN
                expected[k + 1, f"{periodic_col}_gt_count"] = 0
        # special
        for i, special_col in enumerate(info.special_columns):
            expected[0, special_col] = i + 0.5 + 2
            expected[0, f"{special_col}_{info.special_dict[special_col][1]}_count"] = 2 * 3
            expected[0, f"{special_col}_count"] = 2 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{special_col}_count"] = 1 * (k + 1)
        # agg_off
        for i, agg_off_col in enumerate(info.agg_off_columns):
            if i == 0:
                row0 = [-1, 0, 0, 1, None]
            else:
                row0 = [i, i, i, i + 1, None]
            expected = _assign_value(expected, pl.lit(row0), agg_off_col, row=0)
            expected = _assign_value(expected, pl.lit([0]), agg_off_col, row=1)
            expected = _assign_value(expected, pl.lit([0, 0]), agg_off_col, row=2)
        # agg_mean
        for i, agg_mean_col in enumerate(agg_mean_columns):
            expected[0, agg_mean_col] = i + 0.25 + 2
            expected[0, f"{agg_mean_col}_count"] = 3 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{agg_mean_col}_count"] = 1 * (k + 1)
        # count
        expected[0, COLUMN_COUNT] = (n - 1) * 3
        for k in range(len(dfs)):
            expected[k + 1, COLUMN_COUNT] = 1 * (k + 1)
        # time
        expected[0, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        expected[0, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2019, 1, 1, 1, 1, 1)
        for k in range(len(dfs)):
            expected[k + 1, get_agg_column(COLUMN_OCCURRENCE, Stats.MIN)] = datetime.datetime(2019, 1, 1, 1, 0, 0)
            expected[k + 1, get_agg_column(COLUMN_OCCURRENCE, Stats.MAX)] = datetime.datetime(2020, 1, 1, 1, 0, 0)

        # lfs = [df.lazy() for df in dfs]
        lfs = dfs
        actual = merge_quantized_pmw_features(lfs, TC_COLUMNS, agg_off_columns, agg_off_limit=5)  # .collect()
        actual = actual.with_columns([
            pl.col(flag_col).list.sort()
            for flag_col in info.flag_columns
        ])
        actual = actual.sort(by="Tc_10H")

        # TODO should the order be preserved?
        self.assertEqual(sorted(expected.columns), sorted(actual.columns))
        # TODO replace with self.assert... ?
        assert expected.equals(actual.select(expected.columns))


if __name__ == '__main__':
    unittest.main()
