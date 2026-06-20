@author: SaiQU


# one XGBoost model per 0.5deg grid 

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

predictors = [
    'precipitation', 'mean_max_temperature', 'shortwave_radiation', 'co2_concentration',
    'available_water_capacity', 'sand_content', 'ph',
    'elevation', 'slope', 'aspect', 'longitude',
    'cropland_pct', 'forest_pct', 'grassland_pct', 'shrubland_pct', 'bare_pct', 'urban_pct',
]
target = 'kndvi'
grid_col = 'grid_0p5deg_id'

train_path = 'data/kndvi_training_table_2000_2019.csv'
val_path = 'data/kndvi_validation_table_2020.csv'
models_dir = Path('outputs/xgboost_local_models')
models_dir.mkdir(parents=True, exist_ok=True)

xgb_params = dict(objective='reg:squarederror', n_estimators=500, max_depth=6,
                   learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42)

train_df = pd.read_csv(train_path)
print(f'{len(train_df)} pixel-year obs in training set ')

perf = []
for grid_id, grid_df in train_df.groupby(grid_col):
    if len(grid_df) < 30:
        continue  # not enough points for a stable local fit

    X, y = grid_df[predictors], grid_df[target]
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)

    y_pred = model.predict(X)
    perf.append({
        'grid_id': grid_id, 'n_obs': len(grid_df),
        'r2': r2_score(y, y_pred), 'rmse': np.sqrt(mean_squared_error(y, y_pred)),
    })

    with open(models_dir / f'xgb_grid_{grid_id}.pkl', 'wb') as f:
        pickle.dump(model, f)

pd.DataFrame(perf).to_csv('outputs/grid_specific_r2_rmse.csv', index=False)
print('saved per-grid R2/RMSE')

# pooled 2020 check
val_df = pd.read_csv(val_path)
preds = []
for grid_id, grid_df in val_df.groupby(grid_col):
    model_path = models_dir / f'xgb_grid_{grid_id}.pkl'
    if not model_path.exists():
        continue
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    y_pred = model.predict(grid_df[predictors])
    preds.append(pd.DataFrame({'y_true': grid_df[target].values, 'y_pred': y_pred}))

pooled = pd.concat(preds, ignore_index=True)
r2 = r2_score(pooled['y_true'], pooled['y_pred'])
rmse = np.sqrt(mean_squared_error(pooled['y_true'], pooled['y_pred']))
print(f'2020 pooled validation: R2={r2:.3f}, RMSE={rmse:.4f}, n={len(pooled)}')

pd.DataFrame({'r2': [r2], 'rmse': [rmse], 'n': [len(pooled)]}).to_csv(
    'outputs/2020_pooled_validation_summary.csv', index=False)
