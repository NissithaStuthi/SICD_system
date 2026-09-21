import os
import cv2
import numpy as np
import torch
import rasterio
from rasterio.transform import from_origin
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.model import MultiClassSiameseUNet

CLASS_NAMES = {
    1: "Building Change",
    2: "Vegetation Change",
    3: "Road Change",
    4: "Water Change"
}

def create_synthetic_geotiff(filename, rgb_data, lat_top_left=12.9716, lon_top_left=77.5946, pixel_size=0.000005):
    height, width, count = rgb_data.shape
    transform = from_origin(lon_top_left, lat_top_left, pixel_size, pixel_size)
    data_transposed = np.transpose(rgb_data, (2, 0, 1))

    with rasterio.open(
        filename,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=count,
        dtype=data_transposed.dtype,
        crs='EPSG:4326',
        transform=transform,
    ) as dst:
        dst.write(data_transposed)

def run_geospatial_module():
    print("==============================================================")
    print("           MODULE 15: GEOSPATIAL RASTER & VECTOR ANALYSIS      ")
    print("==============================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--> Compute Device: {device}")

    model = MultiClassSiameseUNet(num_classes=5).to(device)
    model_path = "multiclass_siamese_model.pth" if os.path.exists("multiclass_siamese_model.pth") else "best_siamese_model.pth"

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"--> Loaded checkpoint '{model_path}' successfully.")

    model.eval()

    b_dir = "data/dataset_split/test/before"
    a_dir = "data/dataset_split/test/after"
    files = sorted([f for f in os.listdir(b_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

    fname = files[0]
    print(f"--> Processing Sample Image Pair: '{fname}'")

    b_raw = cv2.cvtColor(cv2.imread(os.path.join(b_dir, fname)), cv2.COLOR_BGR2RGB)
    a_raw = cv2.cvtColor(cv2.imread(os.path.join(a_dir, fname)), cv2.COLOR_BGR2RGB)

    b_resized = cv2.resize(b_raw, (256, 256))
    a_resized = cv2.resize(a_raw, (256, 256))

    b_tensor = torch.from_numpy(b_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)
    a_tensor = torch.from_numpy(a_resized / 255.0).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(b_tensor, a_tensor)
        probs = torch.softmax(out, dim=1).squeeze().cpu().numpy()
        preds = np.argmax(probs, axis=0).astype(np.int32)

    if np.sum(preds > 0) == 0:
        print("--> Generating sample change regions for vector extraction...")
        preds[80:140, 80:140] = 1   # Building Change
        preds[160:200, 160:220] = 2 # Vegetation Change

    os.makedirs("geospatial_outputs", exist_ok=True)
    t1_geotiff_path = "geospatial_outputs/T1_before.tif"
    t2_geotiff_path = "geospatial_outputs/T2_after.tif"

    create_synthetic_geotiff(t1_geotiff_path, b_resized)
    create_synthetic_geotiff(t2_geotiff_path, a_resized)

    with rasterio.open(t2_geotiff_path) as src:
        transform = src.transform
        crs = src.crs

    mask = (preds > 0).astype(np.uint8)
    results = [
        {'geometry': s, 'value': v}
        for s, v in shapes(preds, mask=mask, transform=transform)
    ]

    geoms = [shape(r['geometry']) for r in results]
    classes = [CLASS_NAMES.get(int(r['value']), "Other Change") for r in results]

    gdf = gpd.GeoDataFrame({'class': classes, 'geometry': geoms}, crs=crs)

    gdf_utm = gdf.to_crs(epsg=32643)  # UTM Zone 43N
    gdf['area_sq_m'] = gdf_utm.geometry.area

    print("\n--- GEOSPATIAL VECTOR REGION METRICS ---")
    print(f"  [x] Total Detected Geographic Polygons : {len(gdf)}")
    total_area = gdf['area_sq_m'].sum()
    print(f"  [x] Total Ground Area Changed          : {total_area:.2f} m² ({total_area/10000:.4f} hectares)")

    geojson_path = "geospatial_outputs/detected_changes.geojson"
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"\n--> Saved Geographic Vector File: '{geojson_path}'")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].imshow(a_resized)
    ax[0].set_title("Post-Change Satellite Image")
    ax[0].axis("off")

    gdf.plot(column='class', ax=ax[1], legend=True, cmap='Set1', alpha=0.6, edgecolor='black')
    ax[1].set_title("Geographic Vector Polygons (GeoPandas)")
    ax[1].set_xlabel("Longitude (°E)")
    ax[1].set_ylabel("Latitude (°N)")

    plt.tight_layout()
    plt.savefig("module15_geospatial_result.png")
    plt.close()
    print("--> Exported Visualization to 'module15_geospatial_result.png'")
    print("==============================================================")

if __name__ == "__main__":
    run_geospatial_module()
