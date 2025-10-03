import os
import re
import glob
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from shapely.geometry import mapping
import geopandas as gpd
import cv2

def group_files_by_date(raster_folder):
    files = glob.glob(os.path.join(raster_folder, "*_channel_*.tif"))
    dates = {}
    # Pattern :  yyyy_m_d_Andrano_channel_N.tif 
    pat = re.compile(r"^(\d{4}_[0-9]{1,2}_[0-9]{1,2})_Andrano_channel_(\d+)\.tif$")
    for f in files:
        bn = os.path.basename(f)
        m = pat.match(bn)
        if m:
            date_str, ch = m.groups()
            dates.setdefault(date_str, []).append((int(ch), f))
    # trier par numéro de channel
    for d in list(dates.keys()):
        dates[d] = [p for _, p in sorted(dates[d])]
    return dates

def rotate_and_extract_polygon_multichannel(stack, mask, out_path, orig_dtype):
    """
    stack: np.ndarray shape (C, H, W)
    mask: 2D uint8 (0/1 or 0/255) same H,W where polygon==1
    Applique rotation minimale (minAreaRect) sur le mask, applique la même
    transform sur chaque canal, crop minimal et sauvegarde le multibande.
    Retourne cropped_stack (C, h, w).
    """
    H, W = mask.shape
    # normaliser mask uint8 0/255
    mask8 = (mask > 0).astype(np.uint8) * 255

    # findContours: compatibilité OpenCV
    contours_info = cv2.findContours(mask8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_info[0] if len(contours_info) == 2 else contours_info[1]

    if not contours:
        # aucun contour : sauver une pile noire de la taille d'origine
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
    rect = cv2.minAreaRect(largest)   # (center(x,y), (w,h), angle)
    (cx, cy), (rw, rh), angle = rect
    # for convenience, make rw >= rh (rotate 90° if needed)
    if rw < rh:
        angle += 90
        rw, rh = rh, rw

    # préparation rotation
    C, H0, W0 = stack.shape
    image_center = (W0 / 2.0, H0 / 2.0)
    rotation_matrix = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    new_w = int(H0 * abs_sin + W0 * abs_cos)
    new_h = int(H0 * abs_cos + W0 * abs_sin)

    # translation pour garder tout centré
    rotation_matrix[0, 2] += new_w / 2.0 - image_center[0]
    rotation_matrix[1, 2] += new_h / 2.0 - image_center[1]

    # appliquer rotation à chaque canal (on travaille en float32 pour interpolation puis on reconvertit)
    rotated_channels = []
    for i in range(C):
        chan = stack[i].astype(np.float32)
        rotated = cv2.warpAffine(chan, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        rotated_channels.append(rotated)
    rotated_stack = np.stack(rotated_channels, axis=0)  # shape (C, new_h, new_w)

    # calculer la boîte tournée correspondant au rect
    box = cv2.boxPoints(((cx, cy), (rw, rh), angle))  # coordonnées dans l'image originale
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

    if min_x >= max_x or min_y >= max_y:
        black = np.zeros((C, 1, 1), dtype=orig_dtype)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", driver="GTiff",
                           height=1, width=1, count=C, dtype=orig_dtype) as dst:
            for i in range(C):
                dst.write(black[i], i+1)
        return black

    cropped = rotated_stack[:, min_y:max_y, min_x:max_x]
    # reconvertir au dtype d'origine si besoin (attention overflow si conversion incompatible)
    try:
        cropped = cropped.astype(orig_dtype)
    except Exception:
        cropped = cropped.astype(stack.dtype)

    # sauvegarde (NOTE: ici on n'écrit pas de transform géo)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", driver="GTiff",
                       height=cropped.shape[1], width=cropped.shape[2],
                       count=cropped.shape[0], dtype=cropped.dtype) as dst:
        for i in range(cropped.shape[0]):
            dst.write(cropped[i], i+1)

    return cropped



def extract_timeseries(raster_folder,
                                  shapefile_path,
                                  parcel_id=None,
                                  id_column="id",
                                  output_dir="output",
                                  expected_channels=5,
                                  patch_only=True,
                                  full_dataset=False):
    """
    Si full_dataset=True, traite toutes les géométries du shapefile et sauvegarde tout
    directement dans output_dir sous forme de fichiers .tif nommés par parcel_id + date.
    """
    os.makedirs(output_dir, exist_ok=True)
    gdf = gpd.read_file(shapefile_path)
    if id_column not in gdf.columns:
        raise KeyError(f"Colonne '{id_column}' non trouvée dans le shapefile")

    grouped = group_files_by_date(raster_folder)
    if not grouped:
        raise FileNotFoundError(f"Aucun fichier channel trouvé dans {raster_folder}")

    results = {}

    if full_dataset:
        parcels = gdf.itertuples()
    else:
        if parcel_id is None:
            raise ValueError("❌ Vous devez fournir un parcel_id si full_dataset=False")
        parcels = [gdf[gdf[id_column] == parcel_id].iloc[0]]

    for parcel in parcels:
        pid = getattr(parcel, id_column) if full_dataset else parcel_id
        geom = parcel.geometry

        for date_str, channel_files in grouped.items():
            if len(channel_files) != expected_channels:
                print(f"⚠️ {date_str} ignorée : {len(channel_files)} canaux trouvés (attendu {expected_channels})")
                continue

            # ======= bbox pour fenêtrage
            with rasterio.open(channel_files[0]) as ref:
                if gdf.crs != ref.crs:
                    geom_proj = gpd.GeoSeries([geom], crs=gdf.crs).to_crs(ref.crs).iloc[0]
                else:
                    geom_proj = geom
                minx, miny, maxx, maxy = geom_proj.bounds
                row_min, col_min = ref.index(minx, maxy)
                row_max, col_max = ref.index(maxx, miny)
                height = abs(row_max - row_min)
                width = abs(col_max - col_min)
                size = max(width, height)
                row_c = (row_min + row_max) // 2
                col_c = (col_min + col_max) // 2
                half = size // 2
                window = Window(col_c - half, row_c - half, size, size)
                win_transform = ref.window_transform(window)
                shapes = [(mapping(geom_proj), 1)]
                mask = rasterize(shapes,
                                 out_shape=(int(size), int(size)),
                                 transform=win_transform,
                                 fill=0,
                                 all_touched=True,
                                 dtype=np.uint8)

            # lecture des canaux
            patches = []
            for ch_path in channel_files:
                with rasterio.open(ch_path) as src:
                    band = src.read(1, window=window, boundless=True, fill_value=0)
                    patches.append(band)
            stacked = np.stack(patches, axis=0)

            # === sortie (tout dans output_dir)
            out_path = os.path.join(output_dir, f"parcel_{pid}_{date_str}.tif")

            if patch_only:
                cropped_stack = rotate_and_extract_polygon_multichannel(
                    stacked, mask, out_path, orig_dtype=stacked.dtype
                )
                results.setdefault(pid, {})[date_str] = cropped_stack
            else:
                with rasterio.open(channel_files[0]) as ref:
                    out_meta = ref.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": stacked.shape[1],
                        "width": stacked.shape[2],
                        "transform": ref.window_transform(window),
                        "count": stacked.shape[0],
                        "dtype": stacked.dtype
                    })
                    with rasterio.open(out_path, "w", **out_meta) as dst:
                        for i in range(stacked.shape[0]):
                            dst.write(stacked[i], i+1)
                results.setdefault(pid, {})[date_str] = stacked

            print(f"✅ Parcel {pid} | {date_str} -> {out_path}")

    return results
