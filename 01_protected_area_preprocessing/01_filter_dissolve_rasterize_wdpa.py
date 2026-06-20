@author: SaiQU

# filter WDPA to terrestrial PAs, drop tiny polygons, dissolve overlaps, rasterize
 

import geopandas as gpd
import rasterio
from rasterio import features
from shapely.ops import unary_union

wdpa_path = 'data/WDPA_Jan2024_shp/WDPA_Jan2024_polygons.shp'
out_dissolved = 'outputs/wdpa_terrestrial_dissolved.gpkg'
out_raster_fine = 'outputs/pa_mask_0p01667deg.tif'  
out_raster_coarse = 'outputs/pa_mask_0p05deg.tif'    

min_area_m2 = 90000   

gdf = gpd.read_file(wdpa_path)

# drop points, keep polygons only
gdf = gdf[gdf.geometry.type.isin(['Polygon', 'MultiPolygon'])]

# terrestrial only
if 'MARINE' in gdf.columns:
    gdf = gdf[gdf['MARINE'] == 0]

# drop sub-pixel polygons (area in equal-area projection)
gdf_ea = gdf.to_crs('EPSG:6933')
gdf = gdf[gdf_ea.geometry.area >= min_area_m2].copy()

print(f'{len(gdf)} PAs left after filtering (should be ~172009)')

# dissolve everything into one mask, no need to keep individual polygons here
merged = unary_union(gdf.geometry.values)
dissolved = gpd.GeoDataFrame({'pa_id': [1]}, geometry=[merged], crs=gdf.crs)
dissolved.to_file(out_dissolved, driver='GPKG')


def rasterize_mask(gdf, res_deg, out_path):
    minx, miny, maxx, maxy = -180.0, -90.0, 180.0, 90.0
    w = int(round((maxx - minx) / res_deg))
    h = int(round((maxy - miny) / res_deg))
    transform = rasterio.transform.from_bounds(minx, miny, maxx, maxy, w, h)

    shapes = [(geom, 1) for geom in gdf.geometry]
    mask = features.rasterize(shapes, out_shape=(h, w), transform=transform,
                               fill=0, dtype='uint8')

    profile = dict(driver='GTiff', height=h, width=w, count=1, dtype='uint8',
                   crs='EPSG:4326', transform=transform, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mask, 1)
    print('wrote', out_path)


rasterize_mask(dissolved, 1 / 60, out_raster_fine)
rasterize_mask(dissolved, 0.05, out_raster_coarse)
