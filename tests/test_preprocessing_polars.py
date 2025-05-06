import datetime
import unittest
from typing import Dict, List

import numpy as np
import polars as pl

from pmw_analysis.constants import COLUMN_TIME, COLUMN_COUNT, STRUCT_FIELD_COUNT
from pmw_analysis.preprocessing_polars import _round, _get_tc_columns, _get_flag_columns, _get_periodic_columns, \
    _get_special_columns, _get_special_dict, _aggregate, _get_periodic_dict, merge_quantized_pmw_features


def _get_test_columns():
    return [
        'incidenceAngle_LF', 'sunGlintAngle_LF', 'SClatitude', 'Quality_LF', 'sunLocalTime', 'SCorientation', 'lon',
        'lat', 'day', 'incidenceAngle_HF', 'sunGlintAngle_HF', 'Quality_HF', 'Tc_10H', 'Tc_10V', 'Tc_165H',
        'Tc_165V', 'Tc_183V3', 'Tc_183V7', 'Tc_19H', 'Tc_19V', 'Tc_23V', 'Tc_37H', 'Tc_37V', 'Tc_89H', 'Tc_89V',
        'L1CqualityFlag', 'airmassLiftIndex', 'cloudWaterPath', 'convectivePrecipitation', 'frozenPrecipitation',
        'iceWaterPath', 'mostLikelyPrecipitation', 'pixelStatus', 'precip1stTertial', 'precip2ndTertial',
        'precipitationYesNoFlag', 'probabilityOfPrecip', 'qualityFlag', 'rainWaterPath', 'surfacePrecipitation',
        'surfaceTypeIndex', 'temp2mIndex', 'totalColumnWaterVaporIndex'
    ]


def _create_single_row() -> pl.DataFrame:
    columns = _get_test_columns()
    tc_cols = _get_tc_columns(columns)
    flag_cols = _get_flag_columns(columns)
    periodic_cols = _get_periodic_columns(columns)
    special_cols = _get_special_columns(columns)

    tc = {col: 0. for col in tc_cols}
    flag = {col: 0 for col in flag_cols}
    periodic = {col: 0. for col in periodic_cols}
    special = {col: 0. for col in special_cols}
    other = {
        col: 0. for col in columns
        if col not in tc_cols + flag_cols + periodic_cols + special_cols
    }

    df = pl.DataFrame(
        {**tc, **flag, **periodic, **special, **other, COLUMN_TIME: datetime.datetime(2000, 1, 1, 0, 0, 0)})
    return df


def _create_single_row_aggregated() -> pl.DataFrame:
    columns = _get_test_columns()
    tc_cols = _get_tc_columns(columns)
    flag_cols = _get_flag_columns(columns)
    periodic_cols = _get_periodic_columns(columns)
    special_cols = _get_special_columns(columns)
    other_cols = [col for col in columns if col not in tc_cols + flag_cols + periodic_cols + special_cols]

    special_dict = _get_special_dict()

    tc = {col: 0. for col in tc_cols}
    flag = {col: [[]] for col in flag_cols}

    periodic = {k: v
                for col in periodic_cols
                for k, v in {f"{col}_lt": 0., f"{col}_lt_count": 0, f"{col}_gt": 0., f"{col}_gt_count": 0}.items()}

    special = {k: v
               for col in special_cols
               for k, v in {col: 0., f"{col}_{special_dict[col][1]}_count": 0, f"{col}_count": 0}.items()}

    other = {k: v
             for col in other_cols
             for k, v in {col: 0., f"{col}_count": 0.}.items()}

    df = pl.DataFrame(data={
        **tc,
        **flag,
        **periodic,
        **special,
        **special,
        **other,
        COLUMN_TIME: datetime.datetime(2000, 1, 1, 0, 0, 0),
        COLUMN_COUNT: 0,
    }).with_columns([
        pl.col(flag_col)
        .cast(pl.List(pl.Struct([
            pl.Field(flag_col, pl.Int16),
            pl.Field(STRUCT_FIELD_COUNT, pl.Int64)
        ]))) for flag_col in flag_cols
    ])
    return df


def _append_struct(df: pl.DataFrame, structs: List[Dict], column: str, row: int):
    df = df.with_columns(
        pl.when(pl.arange(0, pl.count()) == row)
        .then(pl.col(column).list.concat(pl.lit(structs, dtype=pl.List(pl.Struct))))
        .otherwise(pl.col(column))
    )
    return df

class PreprocessingPolarsTestCase(unittest.TestCase):
    def test_round(self) -> None:
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

        df_after_actual = _round(df_before, tc_columns, uncertainties)

        assert df_after_expected.equals(df_after_actual)


    def test_aggregate(self):
        columns = _get_test_columns()
        tc_cols = _get_tc_columns(columns)
        flag_cols = _get_flag_columns(columns)
        periodic_cols = _get_periodic_columns(columns)
        special_cols = _get_special_columns(columns)
        other_cols = [col for col in columns if col not in tc_cols + flag_cols + periodic_cols + special_cols]

        periodic_dict = _get_periodic_dict()
        special_dict = _get_special_dict()

        #### df before aggregation ####
        n = 6
        df_before = pl.concat([_create_single_row() for _ in range(n)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(n):
                df_before[j, tc_col] = i
        df_before[n - 1, tc_cols[0]] = 1
        # flag
        for i, flag_col in enumerate(flag_cols):
            df_before[0, flag_col] = i
            df_before[1, flag_col] = i
            df_before[2, flag_col] = i
            df_before[3, flag_col] = i + 1
            df_before[4, flag_col] = None
        df_before[0, flag_cols[0]] = -1
        # periodic
        for periodic_col in periodic_cols:
            mid = periodic_dict[periodic_col]
            df_before[0, periodic_col] = mid - 1
            df_before[1, periodic_col] = mid - 1
            df_before[2, periodic_col] = mid
            df_before[3, periodic_col] = mid + 1
            df_before[4, periodic_col] = None
        # special
        for i, special_col in enumerate(special_cols):
            df_before[0, special_col] = i
            df_before[1, special_col] = i + 1
            df_before[2, special_col] = special_dict[special_col][0]
            df_before[3, special_col] = special_dict[special_col][0]
            df_before[4, special_col] = None
        # other
        for i, other_col in enumerate(other_cols):
            df_before[0, other_col] = i
            df_before[1, other_col] = i
            df_before[2, other_col] = i + 1
            df_before[3, other_col] = None
            df_before[4, other_col] = None
        # time
        df_before[0, COLUMN_TIME] = datetime.datetime(2020, 1, 1, 0, 0, 0)
        df_before[1, COLUMN_TIME] = datetime.datetime(2019, 2, 1, 0, 0, 0)
        df_before[2, COLUMN_TIME] = datetime.datetime(2019, 1, 2, 0, 0, 0)
        df_before[3, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        df_before[4, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1, 0, 0)
        df_before[5, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1, 0, 0)

        def append_struct(df: pl.DataFrame, structs: List[Dict], col: str, row: int):
            df = df.with_columns(
                pl.when(pl.arange(0, pl.count()) == row)
                .then(pl.col(col).list.concat(pl.lit(structs, dtype=pl.List(pl.Struct))))
                .otherwise(pl.col(col))
            )
            return df

        #### expected df after aggregation ####
        expected = pl.concat([_create_single_row_aggregated() for _ in range(2)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(2):
                expected[j, tc_col] = i
        expected[1, tc_cols[0]] = 1
        # flag
        for i, flag_col in enumerate(flag_cols):
            if i == 0:
                expected = append_struct(expected,
                                         [{flag_col: None, STRUCT_FIELD_COUNT: 1}, {flag_col: -1, STRUCT_FIELD_COUNT: 1},
                                          {flag_col: i, STRUCT_FIELD_COUNT: 2}, {flag_col: i + 1, STRUCT_FIELD_COUNT: 1}],
                                         flag_col,
                                         0)
            else:
                expected = append_struct(expected,
                                         [{flag_col: None, STRUCT_FIELD_COUNT: 1}, {flag_col: i, STRUCT_FIELD_COUNT: 3},
                                          {flag_col: i + 1, STRUCT_FIELD_COUNT: 1}], flag_col, 0)
            expected = append_struct(expected, [{flag_col: 0, STRUCT_FIELD_COUNT: 1}], flag_col, 1)
        # periodic
        for i, periodic_col in enumerate(periodic_cols):
            expected[0, f"{periodic_col}_lt"] = periodic_dict[periodic_col] - 2 / 3
            expected[0, f"{periodic_col}_lt_count"] = 3
            expected[0, f"{periodic_col}_gt"] = periodic_dict[periodic_col] + 1
            expected[0, f"{periodic_col}_gt_count"] = 1

            expected[1, f"{periodic_col}_lt"] = 0
            expected[1, f"{periodic_col}_lt_count"] = 1
            expected[1, f"{periodic_col}_gt"] = None
            expected[1, f"{periodic_col}_gt_count"] = 0
        # special
        for i, special_col in enumerate(special_cols):
            expected[0, special_col] = i + 0.5
            expected[0, f"{special_col}_{special_dict[special_col][1]}_count"] = 2
            expected[0, f"{special_col}_count"] = 2

            expected[1, f"{special_col}_count"] = 1
        # other
        for i, other_col in enumerate(other_cols):
            expected[0, other_col] = i + 1 / 3
            expected[0, f"{other_col}_count"] = 3

            expected[1, f"{other_col}_count"] = 1
        # time
        expected[0, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        expected[1, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1, 0, 0)
        # count
        expected[0, COLUMN_COUNT] = n - 1
        expected[1, COLUMN_COUNT] = 1

        actual = _aggregate(df_before, tc_cols, flag_cols, periodic_cols, special_cols, [COLUMN_TIME],
                            periodic_dict, special_dict)
        actual = actual.with_columns([
            pl.col(flag_col).list.sort()
            for flag_col in flag_cols
        ])
        actual = actual.sort(by=COLUMN_COUNT, descending=True)

        # TODO should the order be preserved?
        self.assertEqual(sorted(expected.columns), sorted(actual.columns))
        # TODO replace with self.assert... ?
        assert expected.equals(actual.select(expected.columns))


    def test_merge(self):
        columns = _get_test_columns()
        tc_cols = _get_tc_columns(columns)
        flag_cols = _get_flag_columns(columns)
        periodic_cols = _get_periodic_columns(columns)
        special_cols = _get_special_columns(columns)
        other_cols = [col for col in columns if col not in tc_cols + flag_cols + periodic_cols + special_cols]

        periodic_dict = _get_periodic_dict()
        special_dict = _get_special_dict()

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
            for i, flag_col in enumerate(flag_cols):
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
            for i, periodic_col in enumerate(periodic_cols):
                dfs[k][0, f"{periodic_col}_lt"] = periodic_dict[periodic_col] - 1.5 - 3 * k
                dfs[k][0, f"{periodic_col}_lt_count"] = 3 * (k + 1)
                dfs[k][0, f"{periodic_col}_gt"] = periodic_dict[periodic_col] + 1.5 + 3 * k
                dfs[k][0, f"{periodic_col}_gt_count"] = 1 * (k + 1)

                dfs[k][1, f"{periodic_col}_lt"] = 0
                dfs[k][1, f"{periodic_col}_lt_count"] = 1 * (k + 1)
                dfs[k][1, f"{periodic_col}_gt"] = None
                dfs[k][1, f"{periodic_col}_gt_count"] = 0
            # special
            for i, special_col in enumerate(special_cols):
                dfs[k][0, special_col] = i + 0.5 + 3 * k
                dfs[k][0, f"{special_col}_{special_dict[special_col][1]}_count"] = 2 * (k + 1)
                dfs[k][0, f"{special_col}_count"] = 2 * (k + 1)

                dfs[k][1, f"{special_col}_count"] = 1 * (k + 1)
            # other
            for i, other_col in enumerate(other_cols):
                dfs[k][0, other_col] = i + 0.25 + 3 * k
                dfs[k][0, f"{other_col}_count"] = 3 * (k + 1)

                dfs[k][1, f"{other_col}_count"] = 1 * (k + 1)
            # count
            dfs[k][0, COLUMN_COUNT] = (n - 1) * (k + 1)
            dfs[k][1, COLUMN_COUNT] = k + 1
            # time
            dfs[k][0, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1 - k, 0, 0)
            dfs[k][1, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1, 0, 0)

        #### expected df after merging ####
        expected = pl.concat([_create_single_row_aggregated() for _ in range(3)])
        # tc
        for i, tc_col in enumerate(tc_cols):
            for j in range(3):
                expected[j, tc_col] = i
            expected[1, tc_cols[0]] = 1
            expected[2, tc_cols[0]] = 2
        # flag
        for i, flag_col in enumerate(flag_cols):
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
        for i, periodic_col in enumerate(periodic_cols):
            expected[0, f"{periodic_col}_lt"] = periodic_dict[periodic_col] - 1.5 - 2
            expected[0, f"{periodic_col}_lt_count"] = 3 * 3
            expected[0, f"{periodic_col}_gt"] = periodic_dict[periodic_col] + 1.5 + 2
            expected[0, f"{periodic_col}_gt_count"] = 1 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{periodic_col}_lt"] = 0
                expected[k + 1, f"{periodic_col}_lt_count"] = 1 * (k + 1)
                expected[k + 1, f"{periodic_col}_gt"] = np.NaN
                expected[k + 1, f"{periodic_col}_gt_count"] = 0
        # special
        for i, special_col in enumerate(special_cols):
            expected[0, special_col] = i + 0.5 + 2
            expected[0, f"{special_col}_{special_dict[special_col][1]}_count"] = 2 * 3
            expected[0, f"{special_col}_count"] = 2 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{special_col}_count"] = 1 * (k + 1)
        # other
        for i, other_col in enumerate(other_cols):
            expected[0, other_col] = i + 0.25 + 2
            expected[0, f"{other_col}_count"] = 3 * 3

            for k in range(len(dfs)):
                expected[k + 1, f"{other_col}_count"] = 1 * (k + 1)
        # count
        expected[0, COLUMN_COUNT] = (n - 1) * 3
        for k in range(len(dfs)):
            expected[k + 1, COLUMN_COUNT] = 1 * (k + 1)
        # time
        expected[0, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 0, 0, 0)
        for k in range(len(dfs)):
            expected[k + 1, COLUMN_TIME] = datetime.datetime(2019, 1, 1, 1, 0, 0)

        actual = merge_quantized_pmw_features(dfs)
        actual = actual.with_columns([
            pl.col(flag_col).list.sort()
            for flag_col in flag_cols
        ])
        actual = actual.sort(by="Tc_10H")

        # TODO should the order be preserved?
        self.assertEqual(sorted(expected.columns), sorted(actual.columns))
        # TODO replace with self.assert... ?
        assert expected.equals(actual.select(expected.columns))


if __name__ == '__main__':
    unittest.main()
