"""
Example of analyzing signatures that have appeared for the first time later than others.
"""
import pathlib

import polars as pl

from pmw_analysis.analysis.spatial_visualization import plot_variables_on_map
from pmw_analysis.constants import COLUMN_LON, COLUMN_LAT, FILE_DF_FINAL_NEWEST, \
    DIR_NO_SUN_GLINT, DIR_IMAGES, ArgSurfaceType
from pmw_analysis.constants import DIR_PMW_ANALYSIS, TC_COLUMNS, ArgTransform
from pmw_analysis.processing.filter import filter_by_signature_occurrences_count
from pmw_analysis.quantization.script import get_transformation_function
from pmw_analysis.utils.io import combine_paths, file_to_dir


def main():
    arg_transform = ArgTransform.V4
    arg_surface_type = ArgSurfaceType.OCEAN
    df_newest_path = pathlib.Path(
        DIR_PMW_ANALYSIS) / arg_transform.value / arg_surface_type.OCEAN.value / DIR_NO_SUN_GLINT / FILE_DF_FINAL_NEWEST
    transform = get_transformation_function(arg_transform)
    quant_columns = transform(TC_COLUMNS)

    images_dir = combine_paths(path_base=DIR_IMAGES, path_rel=file_to_dir(df_newest_path),
                               path_rel_base=DIR_PMW_ANALYSIS)
    images_dir.mkdir(parents=True, exist_ok=True)

    df_newest = pl.read_parquet(df_newest_path)

    m_occurrences = 1

    df_newest_quant_m, quant_columns_with_suffix = filter_by_signature_occurrences_count(df_newest, m_occurrences,
                                                                                         quant_columns)

    df_newest_m = df_newest.join(df_newest_quant_m, on=quant_columns_with_suffix, how="inner")
    df_newest_m = df_newest_m[quant_columns + [COLUMN_LON, COLUMN_LAT, "L1CqualityFlag", "qualityFlag"]]
    # df_newest_m = df_newest_m[[COLUMN_LON, COLUMN_LAT, "L1CqualityFlag"]]

    m_occurrences_text = "" if m_occurrences == 1 else f"; Signature occurred at least {m_occurrences} times."
    plot_variables_on_map(df_newest_m, arg_transform,
                          images_dir=images_dir,
                          title_text_suffix=m_occurrences_text,
                          file_name_suffix=f"_{m_occurrences}")


if __name__ == '__main__':
    main()
