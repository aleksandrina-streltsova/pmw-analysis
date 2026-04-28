from quantization.aggregation import AggCircularMean, AggValueCounts

from pmw_analysis.constants import COLUMN_LON, COLUMN_L1C_QUALITY_FLAG, VARIABLE_SURFACE_TYPE_INDEX


def build_default_per_column_aggregations() -> dict[str, object]:
    periodic_column_to_range = {
        "sunLocalTime": (0.0, 24.0),
        COLUMN_LON: (-180.0, 180.0),
        "day": (1.0, 366.0),
    }

    flag_column_to_values = {
        "Quality_LF": [-99, 0, 1, 2, 3, None],
        "SCorientation": [-9999, -8000, 0, 180, None],
        "Quality_HF": [-99, 0, 1, 2, 3, None],
        COLUMN_L1C_QUALITY_FLAG: [-10, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, None],
        "airmassLiftIndex": [0, 1, 2, 3, None],
        "pixelStatus": [-99, 0, 1, 2, 3, 4, 5, None],
        "qualityFlag": [-99, 0, 1, 2, 3, None],
        VARIABLE_SURFACE_TYPE_INDEX: [-9999] + list(range(1, 19)) + [None],
    }

    per_column = {
        col: AggValueCounts(values)
        for col, values in flag_column_to_values.items()
    }
    per_column |= {
        col: AggCircularMean(low, high)
        for col, (low, high) in periodic_column_to_range.items()
    }
    return per_column
