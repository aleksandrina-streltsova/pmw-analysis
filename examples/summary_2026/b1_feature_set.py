"""
Run examples/tmp/feature_set.py with its input moved to <base_dir>/partial/final.parquet,
and save the correlation matrix it plots to corr_mtx.csv in the working directory.
"""
import importlib.util
import pathlib
import sys

import polars as pl
import seaborn as sns

PATH_FEATURE_SET = pathlib.Path(__file__).parents[1] / "tmp" / "feature_set.py"

heatmap = sns.heatmap


def _heatmap_and_save(data: pl.DataFrame, **kwargs):
    pl.DataFrame({"feature": data.columns}).hstack(data).write_csv("corr_mtx.csv")
    return heatmap(data, **kwargs)


def main(base_dir: str):
    spec = importlib.util.spec_from_file_location("feature_set", PATH_FEATURE_SET)
    feature_set = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(feature_set)

    # feature_set.main() reads DIR_PMW_ANALYSIS / "partial" / "final.parquet"
    feature_set.DIR_PMW_ANALYSIS = base_dir
    sns.heatmap = _heatmap_and_save
    feature_set.main()


if __name__ == "__main__":
    main(sys.argv[1])
