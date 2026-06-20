@author: SaiQU

# external buffer (same area as the PA itself) around each protected area,
# then clip roads/cities/rivers to the buffer and get distance rasters.
 


import geopandas as gpd
import numpy as np
import rasterio
from rasterio import features
from scipy.ndimage import distance_transform_edt

pa_path = 'outputs/wdpa_terrestrial_dissolved.gpkg'
roads = {
    'highway': 'data/roads/esri_world_roads_highway.shp',
    'primary': 'data/roads/esri_world_roads_primary.shp',
    'secondary': 'data/roads/esri_world_roads_secondary.shp',
    'tertiary': 'data/roads/esri_world_roads_tertiary.shp',
    'local': 'data/roads/esri_world_roads_local.shp',
}
cities_path = 'data/cities/natural_earth_populated_places.shp'
rivers_path = 'data/rivers/rivers_lines.shp'

template_raster = 'outputs/pa_mask_0p01667deg.tif'
out_dir = 'outputs/distance_predictors'

equal_area_crs = 'EPSG:6933'


def equal_area_buffer(pa_gdf):
    # buffer distance d such that area(buffer) - area(pa) == area(pa), solved by bisection
    pa_ea = pa_gdf.to_crs(equal_area_crs).copy()
    pa_ea['pa_area_m2'] = pa_ea.geometry.area

    rings = []
    for _, row in pa_ea.iterrows():
        geom = row.geometry
        target = row['pa_area_m2']
        lo, hi = 0.0, np.sqrt(target / np.pi) * 5
        for _ in range(40):
            mid = (lo + hi) / 2
            ring_area = geom.buffer(mid).area - geom.area
            if ring_area < target:
                lo = mid
            else:
                hi = mid
        d = (lo + hi) / 2
        rings.append(geom.buffer(d).difference(geom))

    return gpd.GeoDataFrame(pa_ea.drop(columns='geometry'), geometry=rings, crs=equal_area_crs)


def remove_overlap(buf_gdf):
    # don't double count overlapping buffer areas between neighbouring PAs
    cleaned = []
    seen = None
    for _, geom in buf_gdf.geometry.items():
        g = geom if seen is None else geom.difference(seen)
        cleaned.append(g)
        seen = g if seen is None else seen.union(g)
    out = buf_gdf.copy()
    out['geometry'] = cleaned
    return out


def rasterize_and_distance(feat_gdf, template_path, out_path):
    with rasterio.open(template_path) as tmpl:
        transform, shape, crs = tmpl.transform, (tmpl.height, tmpl.width), tmpl.crs
        px_deg = transform.a

    feat_gdf = feat_gdf.to_crs(crs)
    shapes = [(g, 1) for g in feat_gdf.geometry if g is not None]
    presence = features.rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype='uint8')

    deg_to_m = 111320.0  # rough, fine for global-scale screening; reproject properly for final product
    dist_m = distance_transform_edt(presence == 0) * px_deg * deg_to_m

    profile = dict(driver='GTiff', height=shape[0], width=shape[1], count=1,
                   dtype='float32', crs=crs, transform=transform, compress='lzw')
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(dist_m.astype('float32'), 1)
    print('wrote', out_path)


pa = gpd.read_file(pa_path)
buf = remove_overlap(equal_area_buffer(pa)).to_crs(pa.crs)

sources = dict(roads)
sources['cities'] = cities_path
sources['rivers'] = rivers_path

for name, path in sources.items():
    feat = gpd.read_file(path)
    clipped = gpd.overlay(feat, buf, how='intersection')
    rasterize_and_distance(clipped, template_raster, f'{out_dir}/distance_to_{name}.tif')
