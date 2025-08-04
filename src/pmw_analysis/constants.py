"""
This module contains constants used throughout the project.
"""

COLUMN_COUNT = "count"
COLUMN_ACCUM_UNIQUE = "accum_unique"
COLUMN_ACCUM_ALL = "accum_all"
COLUMN_OCCURRENCE = "occurrence"
COLUMN_TIME = "time"
COLUMN_TIME_FRACTION = "time_fraction"
COLUMN_SUFFIX_DOMINANT_CYCLE = "_dominant_cycle"
COLUMN_SUFFIX_QUANT = "_quant"
COLUMN_LON = "lon"
COLUMN_LAT = "lat"
COLUMN_LON_BIN = "lon_bin"
COLUMN_LAT_BIN = "lat_bin"
COLUMN_GPM_ID = "gpm_id"
COLUMN_GPM_CROSS_TRACK_ID = "gpm_cross_track_id"
COLUMN_CLUSTER = "cluster"
COLUMN_L1C_QUALITY_FLAG = "L1CqualityFlag"

COLUMN_OCCURRENCE_TIME = f"occurrence_{COLUMN_TIME}"
COLUMN_OCCURRENCE_LON = f"occurrence_{COLUMN_LON}"
COLUMN_OCCURRENCE_LAT = f"occurrence_{COLUMN_LAT}"

STRUCT_FIELD_COUNT = "struct_count"

ATTR_NAME = "name"
ATTR_PERIODOGRAM_DICT = "periodogram_dict"

DIM_ALONG_TRACK = "along_track"
DIM_CROSS_TRACK = "cross_track"
DIM_PMW = "pmw_frequency"

SAVEFIG_FLAG = True
DEBUG_FLAG = False

SAVEFIG_DIR = "images"
BUCKET_DIR = "/ltenas8/data/GPM_Buckets/GMI"
PMW_ANALYSIS_DIR = "/ltenas8/data/PMW_Analysis"

PRODUCT_1C_GMI_R = "1C-GMI-R"
PRODUCT_2A_GMI = "2A-GMI"
PRODUCT_TYPE_RS = "RS"

VARIABLE_TC = "Tc"
VARIABLE_SURFACE_TYPE_INDEX = "surfaceTypeIndex"

VERSION = 7

STORAGE_GES_DISC = "GES_DISC"

TC_COLUMNS = ["Tc_10H", "Tc_10V",
              "Tc_19H", "Tc_19V",
              "Tc_23V",
              "Tc_37H", "Tc_37V",
              "Tc_89H", "Tc_89V",
              "Tc_165H", "Tc_165V",
              "Tc_183V3", "Tc_183V7"]

AGG_OFF_COLUMNS = [COLUMN_LON, COLUMN_LAT,
                   COLUMN_GPM_ID, COLUMN_GPM_CROSS_TRACK_ID]

ST_COLUMNS = ['Ocean',
              'Sea-Ice',
              'High vegetation',
              'Medium vegetation',
              'Low vegetation',
              'Sparse vegetation',
              'Desert',
              'Elevated snow cover',
              'High snow cover',
              'Moderate snow cover',
              'Light snow cover',
              'Standing Water',
              'Ocean or water Coast',
              'Mixed land/ocean or water coast',
              'Land coast',
              'Sea-ice edge',
              'Mountain rain',
              'Mountain snow']

ST_GROUP_OCEAN = ["Ocean"]
ST_GROUP_VEGETATION = ['High vegetation',
                       'Medium vegetation',
                       'Low vegetation',
                       'Mountain rain']
ST_GROUP_SNOW = ['Elevated snow cover',
                 'High snow cover',
                 'Moderate snow cover',
                 'Mountain snow']
ST_GROUP_EDGES = ['Ocean or water Coast',
                  'Mixed land/ocean or water coast',
                  'Land coast',
                  'Sea-ice edge']
ST_GROUP_MISC = ['Sea-Ice',
                 'Sparse vegetation',
                 'Desert',
                 'Light snow cover',
                 'Standing Water']

EXTENT_GREENLAND = [-73, -11, 59, 83]

AGG_OFF_LIMIT = 100
