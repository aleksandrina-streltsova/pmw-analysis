import pathlib
import polars as pl
import matplotlib.pyplot as plt

from pmw_analysis.constants import TC_COLUMNS, DIR_PMW_ANALYSIS, COLUMN_COUNT, ArgTransform
from pmw_analysis.quantization.script import get_transformation_function


def main():
    args_transform = ArgTransform.V2
    transform = get_transformation_function(args_transform)
    feature_columns = transform(TC_COLUMNS)
    path = pathlib.Path(DIR_PMW_ANALYSIS) / args_transform / "final.parquet"

    df = pl.read_parquet(path)
    for tc_col in feature_columns:
        fig, axes = plt.subplots(2, 1, figsize=(10,5))
        x_min = None
        x_max = None

        for i, y_scale in enumerate(["linear", "log"]):
            ax = axes[i]
            title = args_transform
            bins = 40
            value_count = df[[tc_col, COLUMN_COUNT]].group_by(tc_col).sum()
            value_count = value_count.filter(pl.col(tc_col).is_not_null())

            if x_min is None:
                x_min = value_count[tc_col].min()
                x_max = value_count[tc_col].max()

            ax.hist(value_count[tc_col], weights=value_count[COLUMN_COUNT], bins=bins)
            ax.set_xlabel(tc_col)
            ax.set_ylabel(COLUMN_COUNT)
            ax.set_yscale(y_scale)
            ax.set_title(f"{tc_col} {title}")
            ax.set_xlim(x_min, x_max)
        fig.tight_layout()
        fig.show()

if __name__ == '__main__':
    main()