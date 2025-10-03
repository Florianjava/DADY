import os
import re
import numpy as np
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping
import geopandas as gpd
import cv2
from pathlib import Path

def list_rasters_for_parcel(raster_folder, parcel_code):
    files = sorted([f for f in os.listdir(raster_folder) if f.endswith(f"_{parcel_code}.tif")])
    return files


def rotate_and_extract_polygon_multichannel(stack, mask, out_path, orig_dtype):
    """
    stack: np.ndarray shape (C, H, W)
    mask: 2D uint8 (0/1) same H,W where polygon==1
    Applique rotation minimale (minAreaRect) sur le mask, applique la même
    transform sur chaque canal, crop minimal et sauvegarde le multibande.
    """
    H, W = mask.shape
    mask8 = (mask > 0).astype(np.uint8) * 255
    findres = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = findres[0] if len(findres) == 2 else findres[1]

    if not contours:
        print("⚠️ Aucun contour trouvé dans le mask : on écrit une pile noire.")
        black = np.zeros_like(stack, dtype=orig_dtype)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", driver="GTiff",
                           height=black.shape[1], width=black.shape[2],
                           count=black.shape[0], dtype=black.dtype) as dst:
            for i in range(black.shape[0]):
                dst.write(black[i], i+1)
        return black

    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    (cx, cy), (rw, rh), angle = rect
    if rw < rh:
        angle += 90
        rw, rh = rh, rw

    C, H0, W0 = stack.shape
    image_center = (W0 / 2.0, H0 / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(H0 * abs_sin + W0 * abs_cos)
    new_h = int(H0 * abs_cos + W0 * abs_sin)

    rotation_matrix[0, 2] += new_w / 2.0 - image_center[0]
    rotation_matrix[1, 2] += new_h / 2.0 - image_center[1]

    rotated_channels = []
    for i in range(C):
        chan = stack[i].astype(np.float32)
        rotated = cv2.warpAffine(chan, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_channels.append(rotated)
    rotated_stack = np.stack(rotated_channels, axis=0)

    box = cv2.boxPoints(((cx, cy), (rw, rh), angle))
    ones = np.ones((4, 1))
    pts = np.hstack([box, ones])
    rotated_box = rotation_matrix.dot(pts.T).T
    rotated_box = np.intp(rotated_box)

    x_coords = rotated_box[:, 0]
    y_coords = rotated_box[:, 1]
    min_x = max(0, int(x_coords.min()))
    max_x = min(rotated_stack.shape[2], int(x_coords.max()))
    min_y = max(0, int(y_coords.min()))
    max_y = min(rotated_stack.shape[1], int(y_coords.max()))

    cropped = rotated_stack[:, min_y:max_y, min_x:max_x]
    cropped = cropped.astype(orig_dtype)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=cropped.shape[1], width=cropped.shape[2],
                       count=cropped.shape[0], dtype=cropped.dtype) as dst:
        for i in range(cropped.shape[0]):
            dst.write(cropped[i], i+1)

    return cropped


def extract_timeseries(raster_folder,
                            shapefile_folder,
                            tree_id=None,
                            output_dir="output",
                            patch_only=True,
                            full_dataset=False):
    """
    Extrait pour un arbre ou pour tout le dossier :
     - lit les 5 premiers canaux d’une ortho multibande
     - prend les shapefiles {tree_id}A.shp, {tree_id}B.shp, {tree_id}C.shp
     - full_dataset=True => parcourt tous les arbres dans shapefile_folder
     - résultat = crop autour du polygone (pas full image !)
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- détecter les arbres
    if full_dataset:
        shp_files = [f for f in os.listdir(shapefile_folder) if f.endswith(".shp")]
        tree_set = set()
        for f in shp_files:
            name = os.path.splitext(f)[0]
            m = re.match(r"^(P\d{2}A\d+)[A-C]$", name)
            if m:
                tree_set.add(m.group(1))
        tree_ids = sorted(tree_set)
        if not tree_ids:
            raise FileNotFoundError("Aucun arbre trouvé dans les shapefiles.")
    else:
        if tree_id is None:
            raise ValueError("Si full_dataset=False, il faut fournir tree_id (ex: 'P01A1').")
        tree_ids = [tree_id]

    results = {}

    for tid in tree_ids:
        print(f"\n➡️ Traitement pour {tid}")
        poly_files = [os.path.join(shapefile_folder, f"{tid}{letter}.shp") for letter in ["A", "B", "C"]]
        if any(not os.path.exists(p) for p in poly_files):
            print(f"⚠️ Shapefiles manquants pour {tid}, on saute")
            continue

        results[tid] = {}
        parcel_code = tid[:3]
        raster_files = list_rasters_for_parcel(raster_folder, parcel_code)
        if not raster_files:
            print(f"⚠️ Aucun raster trouvé pour {parcel_code}")
            continue

        for raster_fname in raster_files:
            date_str = os.path.splitext(raster_fname)[0].split("_")[0]
            raster_path = os.path.join(raster_folder, raster_fname)
            print(f"  • date {date_str} -> {raster_fname}")

            with rasterio.open(raster_path) as src:
                bands = src.read(list(range(1, min(6, src.count+1))))  # 5 premiers canaux
                orig_dtype = bands.dtype
                results[tid].setdefault(date_str, {})

                for poly_shp in poly_files:
                    poly_letter = os.path.splitext(os.path.basename(poly_shp))[0][-1]
                    gdf = gpd.read_file(poly_shp)
                    geom = gdf.geometry.iloc[0]
                    if gdf.crs != src.crs:
                        geom = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(src.crs).iloc[0]

                    # Fenêtre carrée autour du polygone
                    minx, miny, maxx, maxy = geom.bounds
                    row_min, col_min = src.index(minx, maxy)
                    row_max, col_max = src.index(maxx, miny)
                    h = abs(row_max - row_min)
                    w = abs(col_max - col_min)
                    size = max(1, max(h, w))
                    row_c = (row_min + row_max) // 2
                    col_c = (col_min + col_max) // 2
                    half = size // 2
                    window = rasterio.windows.Window(col_c - half, row_c - half, size, size)
                    win_transform = src.window_transform(window)

                    # lire fenêtre
                    window_arr = src.read(list(range(1, min(6, src.count+1))),
                                          window=window, boundless=True, fill_value=0)

                    # mask fenêtré
                    mask_window = rasterize(
                        [(mapping(geom), 1)],
                        out_shape=(size, size),
                        transform=win_transform,
                        fill=0,
                        all_touched=True,
                        dtype=np.uint8
                    )

                    out_path = os.path.join(output_dir, f"{tid}_{date_str}_{poly_letter}.tif")
                    if patch_only:
                        cropped = rotate_and_extract_polygon_multichannel(window_arr, mask_window, out_path, orig_dtype)
                        results[tid][date_str][poly_letter] = cropped
                    else:
                        meta = src.meta.copy()
                        meta.update({
                            "driver": "GTiff",
                            "height": window_arr.shape[1],
                            "width": window_arr.shape[2],
                            "transform": win_transform,
                            "count": window_arr.shape[0],
                            "dtype": window_arr.dtype
                        })
                        with rasterio.open(out_path, "w", **meta) as dst:
                            for b in range(window_arr.shape[0]):
                                dst.write(window_arr[b], b+1)
                        results[tid][date_str][poly_letter] = window_arr

                    print(f"    ✅ sauvegardé : {out_path}")

    return results
