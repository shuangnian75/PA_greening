@author: SaiQU


# pairwise pearson correlation among the 20 candidate predictors, drop
# anything with |r|>0.75 
 

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

predictors = [
    'precipitation', 'mean_max_temperature', 'shortwave_radiation', 'co2_concentration',
    'available_water_capacity', 'clay_content', 'silt_content', 'sand_content', 'ph',
    'elevation', 'slope', 'aspect', 'latitude', 'longitude',
    'cropland_pct', 'forest_pct', 'grassland_pct', 'shrubland_pct', 'bare_pct', 'urban_pct',
]
threshold = 0.75

training_table = 'data/kndvi_training_table_2000_2019.csv'
out_fig = 'outputs/extended_data_fig6_predictor_correlation.png'
out_selected = 'outputs/selected_predictors_17.txt'

df = pd.read_csv(training_table)
corr = df[predictors].corr(method='pearson')

remaining = list(predictors)
while True:
    sub = corr.loc[remaining, remaining]
    bad_pairs = [(i, j) for i in remaining for j in remaining if i != j and abs(sub.loc[i, j]) > threshold]
    if not bad_pairs:
        break
    counts = pd.Series([p[0] for p in bad_pairs]).value_counts()
    worst = counts.idxmax()
    remaining.remove(worst)
    print('dropping', worst, '- too correlated with others (>', threshold, ')')

print(f'\n{len(remaining)} predictors left:', remaining)
with open(out_selected, 'w') as f:
    f.write('\n'.join(remaining))

plt.figure(figsize=(10, 9))
sns.heatmap(corr, cmap='RdBu_r', center=0, vmin=-1, vmax=1, square=True, cbar_kws={'label': 'Pearson r'})
plt.title('predictor correlation matrix  ')
plt.tight_layout()
plt.savefig(out_fig, dpi=300)
print('saved figure to', out_fig)
