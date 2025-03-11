"""
Example of reading from bucket.
"""
import gpm
import matplotlib.pyplot as plt
import polars as pl
from gpm.bucket import LonLatPartitioning

# Define extent and bucket directory
extent = [28, 32, 58, 62]
BUCKET_DIR = "/home/rina/Desktop/data_spb"

####--------------------------------------------------------------------------.
#### Define analysis options
# - GMI: 10.6 GHz, 18.7 GHz

LON_COLUMN = 'lon'
LAT_COLUMN = 'lat'
# TODO: what do different columns mean?
QUALITY_LF_COL = 'Quality_LF'
PRECIP_PROB_COL = "probabilityOfPrecip"
SURFACE_PRECIP_COL = 'surfacePrecipitation'

columns = [
    LON_COLUMN,
    LAT_COLUMN,
    'time',

    'Tc_10H',
    'Tc_10V',
    'Tc_89H',

    'L1CqualityFlag',
    QUALITY_LF_COL,
    'Quality_HF',
    'qualityFlag',
    'pixelStatus',

    'precipitationYesNoFlag',
    PRECIP_PROB_COL,

    SURFACE_PRECIP_COL,
]

# -----------------------------------------------------------------------------.
#### Read Parquet Dataset with polars
df_pl: pl.DataFrame = gpm.bucket.read(bucket_dir=BUCKET_DIR,
                                      columns=columns,
                                      use_pyarrow=False,  # use rust parquet reader
                                      extent=extent,
                                      # n_rows=2,
                                      # n_rows=100_000_000, # for prototyping
                                      parallel="auto",  # "row_groups", "columns"
                                      )

# TODO: preprocess nulls
df_pl = df_pl.filter(~pl.col("Tc_10H").is_null())

df_pl_dry = df_pl.filter(pl.col("surfacePrecipitation").is_null())
df_pl_rainy = df_pl.filter(~pl.col("surfacePrecipitation").is_null())
print(df_pl_dry.shape)
print(df_pl_rainy.shape)

df_pl_subset = df_pl_dry

print(df_pl["precipitationYesNoFlag"].sum() / df_pl["precipitationYesNoFlag"].shape[0])

for col in [PRECIP_PROB_COL, QUALITY_LF_COL]:
    plt.hist(df_pl[col])
    plt.show()

# -----------------------------------------------------------------------------.
#### Define LonLatPartitioning
# paritioning ---> Pass extent of country/continent ...
# 0.01 is about 1 km
# 0.05 is about 5 km

partitioning = LonLatPartitioning(size=0.05, extent=extent)

# -----------------------------------------------------------------------------.
#### Group by partititions
df_pl_subset = partitioning.add_labels(df_pl_subset, x="lon", y="lat")
df_pl_subset = partitioning.add_centroids(df_pl_subset, x="lon", y="lat",
                                          x_coord="lon_bin", y_coord="lat_bin")
grouped_df = df_pl_subset.group_by(partitioning.levels)

####---------------------------------------------------------------------------.
#### - Compute Min, Mean, Max Statistics
list_variables = [
    'Tc_10H',
    'Tc_10V',
]

list_expressions = (
        [pl.col(variable).min().name.prefix("min_") for variable in list_variables] +
        [pl.col(variable).max().name.prefix("max_") for variable in list_variables] +
        [pl.col(variable).mean().name.prefix("mean_") for variable in list_variables] +
        [pl.col(variable).median().name.prefix("median_") for variable in list_variables] +
        [pl.col("Tc_10H").count().alias("count")]
)

df_agg = grouped_df.agg(*list_expressions)

####--------------------------------------------------------------------------.
#### Conversion to xarray
# df = partitioning.add_centroids(df, x="lon", y="lat")
# ds = partitioning.to_xarray(df)
# ds = ds.rename({"lon_c": "longitude", "lat_c": "latitude"})

ds = partitioning.to_xarray(df_agg, spatial_coords=("lon_bin", "lat_bin"))
ds = ds.rename({"lon_bin": "longitude", "lat_bin": "latitude"})

####--------------------------------------------------------------------------.
#### Plot RFI
# - Reflected RFI from the earth surface at 10 GHz from direct broadcast satellites

# City: Saint Petersburg
# Define point with RFI
point_city = (30.36, 59.93)
# Define point without RFI
point_outskirts = (31.05, 58.95)

# Define figure kwargs
fig_kwargs = {"figsize": (6, 7), "dpi": 300}

# Define colorbar settings
cbar_kwargs = {"extend": "both",
               "extendfrac": 0.05,
               "label": "Brightness Temperature [K]"}

# Plot RFI pattern (X band)
p = ds["max_Tc_10H"].gpm.plot_map(x="longitude", y="latitude", cmap="Spectral_r",
                                  vmin=260, vmax=350,
                                  fig_kwargs=fig_kwargs,
                                  cbar_kwargs=cbar_kwargs)
p.axes.set_title("Maximum GMI Tb at 10 GHz")
p.axes.scatter(point_city[0], point_city[1], marker="x", c="black")
p.axes.scatter(point_outskirts[0], point_outskirts[1], marker="x", c="blue")
plt.show()

# Plot RFI pattern (X band)
p = ds["median_Tc_10H"].gpm.plot_map(x="longitude", y="latitude", cmap="Spectral_r",
                                     # vmin=260, vmax=350,
                                     fig_kwargs=fig_kwargs,
                                     cbar_kwargs=cbar_kwargs)
p.axes.set_title("Maximum GMI Tb at 10 GHz")
p.axes.scatter(point_city[0], point_city[1], marker="x", c="black")
p.axes.scatter(point_outskirts[0], point_outskirts[1], marker="x", c="blue")
plt.show()
