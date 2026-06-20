
@author: SaiQU
 

# pixel-level Sen's slope (kNDVI) + Mann-Kendall p-value,
# applied to each scenario).

from pathlib import Path

import numpy as np
import pandas as pd
import pymannkendall as mk

years = list(range(2020, 2101, 5))
scenarios = ['SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5']

ts_tpl = 'outputs/kndvi_projections/{scenario}/pixel_timeseries.csv'   
cropland_change_path = 'outputs/lucc_simulation/cropland_fraction_change_2020_2100.csv'   
out_tpl = 'outputs/statistics/{scenario}_pixel_trend_table.csv'


def sen_slope_decade(values):
    valid = ~np.isnan(values)
    if valid.sum() < 4:
        return np.nan, np.nan
    res = mk.original_test(values[valid])
    return res.slope * 10, res.p  # annual -> per decade


def intensity_class(pct):
    if pct <= 30:
        return 'low'
    elif pct <= 60:
        return 'moderate'
    return 'high'


def decline_class(slope, t1, t2):
    if slope >= 0 or np.isnan(slope):
        return None
    if slope > t1:
        return 'weak'
    elif slope > t2:
        return 'moderate'
    return 'severe'


year_cols = [str(y) for y in years]
all_dfs = []

for scenario in scenarios:
    ts = pd.read_csv(ts_tpl.format(scenario=scenario))

    slopes, pvals = [], []
    for _, row in ts.iterrows():
        s, p = sen_slope_decade(row[year_cols].values.astype(float))
        slopes.append(s)
        pvals.append(p)
    ts['sen_slope_decade'] = slopes
    ts['mk_pvalue'] = pvals
    ts['scenario'] = scenario

    cropland = pd.read_csv(cropland_change_path)
    cropland = cropland[cropland['scenario'] == scenario]
    ts = ts.merge(cropland[['pixel_id', 'cropland_expansion_pct']], on='pixel_id', how='left')
    ts['cropland_intensity_class'] = ts['cropland_expansion_pct'].apply(intensity_class)
    ts['cropland_intensity_ordinal'] = ts['cropland_intensity_class'].map({'low': 1, 'moderate': 2, 'high': 3})

    all_dfs.append(ts)

pooled = pd.concat(all_dfs, ignore_index=True)

# tercile breaks from negative slopes pooled across all SSPs, then applied to each scenario
neg = pooled.loc[pooled['sen_slope_decade'] < 0, 'sen_slope_decade']
t1, t2 = np.nanpercentile(neg, [66.67, 33.33])
print(f'tercile breaks: t1={t1:.5f}, t2={t2:.5f}')

pooled['decline_severity'] = pooled['sen_slope_decade'].apply(lambda s: decline_class(s, t1, t2))

for scenario in scenarios:
    out_path = Path(out_tpl.format(scenario=scenario))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pooled[pooled['scenario'] == scenario].to_csv(out_path, index=False)
    print('wrote', out_path)
