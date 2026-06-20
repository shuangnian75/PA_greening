@author: SaiQU

# resample the 18 CMIP6 GCMs to a common 0.05deg grid and take the ensemble
# mean 
# r1i1p1f1 member used where available

from pathlib import Path

import numpy as np
import xarray as xr
import xesmf as xe

gcms = [
    'ACCESS-CM2', 'ACCESS-ESM1-5', 'BCC-CSM2-MR', 'CanESM5', 'CESM2',
    'CMCC-ESM2', 'CNRM-CM6-1', 'EC-Earth3', 'GFDL-ESM4', 'INM-CM5-0',
    'IPSL-CM6A-LR', 'KACE-1-0-G', 'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR',
    'MRI-ESM2-0', 'NorESM2-MM', 'UKESM1-0-LL',
]
scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
variables = {'tasmax': 'mean_max_temperature', 'pr': 'precipitation', 'rsds': 'shortwave_radiation'}

raw_dir = Path('data/cmip6_raw')  # {gcm}/{scenario}/{var}.nc
res = 0.05
out_dir = Path('outputs/cmip6_ensemble')
out_dir.mkdir(parents=True, exist_ok=True)

lon = np.arange(-180 + res / 2, 180, res)
lat = np.arange(-90 + res / 2, 90, res)
target_grid = xr.Dataset({'lat': (['lat'], lat), 'lon': (['lon'], lon)})


def regrid(ds):
    rg = xe.Regridder(ds, target_grid, method='bilinear', periodic=True)
    return rg(ds)


for var, var_name in variables.items():
    for scenario in scenarios:
        members = []
        for gcm in gcms:
            f = raw_dir / gcm / scenario / f'{var}.nc'
            if not f.exists():
                print('missing', f)
                continue
            ds = xr.open_dataset(f)
            if 'member_id' in ds.dims:
                ds = ds.sel(member_id='r1i1p1f1') if 'r1i1p1f1' in ds['member_id'].values else ds.isel(member_id=0)
            members.append(regrid(ds[[var]])[var].assign_coords(gcm=gcm))

        if not members:
            print('no data for', var, scenario, '- skipping')
            continue

        stacked = xr.concat(members, dim='gcm')
        stacked.mean(dim='gcm').to_netcdf(out_dir / f'{var_name}_{scenario}_ensemble_mean.nc')
        stacked.std(dim='gcm').to_netcdf(out_dir / f'{var_name}_{scenario}_ensemble_std.nc')
        print('done', var_name, scenario)
