# Define analysis time period
import datetime

import gpm
import matplotlib.pyplot as plt
import xarray as xr

from pmw_analysis.constants import ALONG_TRACK_DIM

# Define product
PRODUCT = "1C-GMI-R"
PRODUCT_TYPE = "RS"
VARIABLE_TC = "Tc"
DIM_PMW = "pmw_frequency"
VERSION = 7

STORAGE = "GES_DISC"

dates = [
    datetime.date(2015, 3, 20),
    datetime.date(2015, 6, 21),
    datetime.date(2015, 9, 23),
    datetime.date(2015, 12, 22),
]

ds_list = []

for date in dates:
    start_time = datetime.datetime(date.year, date.month, date.day, hour=0, minute=0, second=0)
    end_time = start_time + datetime.timedelta(days=1)

    gpm.download(PRODUCT, start_time, end_time, PRODUCT_TYPE, VERSION, STORAGE)

    dt = gpm.open_datatree(PRODUCT, start_time, end_time, VARIABLE_TC, product_type=PRODUCT_TYPE, chunks={})
    ds = dt.gpm.regrid_pmw_l1(scan_mode_reference="S1")

    ds_list.append(ds)

ds = xr.concat(ds_list, dim=ALONG_TRACK_DIM)

da: xr.DataArray = ds[VARIABLE_TC]
# accessor: GPM_DataArray_Accessor
# accessor.plot_map(add_swath_lines=False)
da.sel(pmw_frequency="19H").gpm.plot_map(add_swath_lines=False)
plt.show()
