@author: SaiQU

 

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

predictors = [
    'precipitation', 'mean_max_temperature', 'shortwave_radiation', 'co2_concentration',
    'available_water_capacity', 'sand_content', 'ph',
    'elevation', 'slope', 'aspect', 'longitude',
    'cropland_pct', 'forest_pct', 'grassland_pct', 'shrubland_pct', 'bare_pct', 'urban_pct',
]
grid_col = 'grid_0p5deg_id'
pid = 'pixel_id'
models_dir = Path('outputs/xgboost_local_models')

scenario = 'SSP3-7.0'  
year = 2050

d1_t_path = f'data/future_predictors/{scenario}/{year}_predictors.csv'
d1_2020_path = f'data/future_predictors/{scenario}/2020_predictors.csv'
d2_t_path = f'outputs/counterfactual_predictors/{scenario}/{year}_D2_cropland_fixed.csv'
d3_t_path = f'outputs/counterfactual_predictors/{scenario}/{year}_D3_all_lucc_fixed.csv'
d3_2020_path = f'outputs/counterfactual_predictors/{scenario}/2020_D3_all_lucc_fixed.csv'

out_path = f'outputs/attribution/{scenario}/{year}_attribution.csv'


def predict(df):
    preds = pd.Series(index=df.index, dtype=float)
    for grid_id, grid_df in df.groupby(grid_col):
        model_path = models_dir / f'xgb_grid_{grid_id}.pkl'
        if not model_path.exists():
            continue
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        preds.loc[grid_df.index] = model.predict(grid_df[predictors])
    return preds


d1_t = pd.read_csv(d1_t_path)
d1_2020 = pd.read_csv(d1_2020_path)
d2_t = pd.read_csv(d2_t_path)
d3_t = pd.read_csv(d3_t_path)
d3_2020 = pd.read_csv(d3_2020_path)

D1_tk = predict(d1_t)
D1_2020 = predict(d1_2020)
D2_tk = predict(d2_t)
D3_tk = predict(d3_t)
D3_2020 = predict(d3_2020)

delta_total = D1_tk - D1_2020                  
delta_cropland = D1_tk - D2_tk                  
delta_lucc = D1_tk - D3_tk                       
delta_noncrop = delta_lucc - delta_cropland       
delta_clim_co2 = D3_tk - D3_2020                 
delta_res = delta_total - (delta_cropland + delta_noncrop + delta_clim_co2) 

result = pd.DataFrame({
    pid: d1_t[pid],
    'scenario': scenario, 'year': year,
    'delta_kndvi_total': delta_total,
    'delta_kndvi_cropland': delta_cropland,
    'delta_kndvi_lucc': delta_lucc,
    'delta_kndvi_noncrop': delta_noncrop,
    'delta_kndvi_clim_co2': delta_clim_co2,
    'delta_kndvi_residual_check': delta_res,
})

print('max |residual| (should be ~0):', np.nanmax(np.abs(result['delta_kndvi_residual_check'])))

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
result.to_csv(out_path, index=False)
print('wrote', out_path)
