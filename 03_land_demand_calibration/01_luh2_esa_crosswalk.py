
@author: SaiQU


# crosswalk between LUH2 classes and ESA-CCI classes -> 7 harmonized classes
 


import pandas as pd

luh2_to_class = {
    'c3ann': 'cropland', 'c4ann': 'cropland', 'c3per': 'cropland',
    'c4per': 'cropland', 'c3nfx': 'cropland',
    'primf': 'forest', 'secdf': 'forest',
    'pastr': 'grassland', 'range': 'grassland',
    'secdn': 'shrubland',
    'urban': 'urban land',
    # water held at baseline area, not a dynamic LUH2 class
}

esa_to_class = {
    10: 'cropland', 11: 'cropland', 12: 'cropland', 20: 'cropland',
    50: 'forest', 60: 'forest', 61: 'forest', 62: 'forest', 70: 'forest',
    71: 'forest', 72: 'forest', 80: 'forest', 81: 'forest', 82: 'forest',
    90: 'forest', 100: 'forest',
    110: 'grassland', 130: 'grassland',
    120: 'shrubland', 121: 'shrubland', 122: 'shrubland',
    140: 'bare land', 150: 'bare land', 152: 'bare land', 153: 'bare land',
    200: 'bare land', 201: 'bare land', 202: 'bare land',
    190: 'urban land',
    210: 'water', 220: 'water',
}


def harmonise_luh2(df, col='luh2_class'):
    df = df.copy()
    df['harmonised_class'] = df[col].map(luh2_to_class)
    missing = df.loc[df['harmonised_class'].isna(), col].unique()
    if len(missing):
        print('unmapped LUH2 classes, add these to luh2_to_class:', missing)
    return df


def harmonise_esa(df, col='esa_cci_code'):
    df = df.copy()
    df['harmonised_class'] = df[col].map(esa_to_class)
    missing = df.loc[df['harmonised_class'].isna(), col].unique()
    if len(missing):
        print('unmapped ESA-CCI codes, add these to esa_to_class:', missing)
    return df


if __name__ == '__main__':
  
    print(harmonise_luh2(pd.DataFrame({'luh2_class': ['c3ann', 'primf', 'pastr', 'urban']})))
    print(harmonise_esa(pd.DataFrame({'esa_cci_code': [10, 50, 110, 190, 210]})))
