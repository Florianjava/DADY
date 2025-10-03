import rasterio
import numpy as np
import os
import random

# === CONFIGURATION ===
INPUT_FOLDER = r"D:\Mes Donnees\dady_data_1\data\UC2_faidherbia\orthoimages_MS_2021"
OUTPUT_FOLDER = "224x224_patchs"
PATCH_SIZE = 224
NB_PATCHES = 10
MAX_TRIES = 1000
PIXEL_NOIR = 0
PIXEL_BLANC = 65535

def is_valid_patch(patch):
    """Vérifie que le patch ne contient pas de pixels noirs ou blancs."""
    return not ((patch == PIXEL_NOIR).any() or (patch == PIXEL_BLANC).any())

def get_random_valid_patch(raster):
    """Retourne un patch valide de taille 224x224 sans noir ni blanc."""
    _, height, width = raster.shape
    for _ in range(MAX_TRIES):
        x = random.randint(0, width - PATCH_SIZE)
        y = random.randint(0, height - PATCH_SIZE)
        patch = raster[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        if patch.shape[1:] == (PATCH_SIZE, PATCH_SIZE) and is_valid_patch(patch):
            return patch, x, y
    raise RuntimeError("Aucun patch valide trouvé.")

def save_patch(path, patch, transform, crs):
    """Sauvegarde un patch de 5 canaux au format GeoTIFF 8-bit."""
    patch = patch.astype(np.uint8)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=PATCH_SIZE,
        width=PATCH_SIZE,
        count=5,
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(5):
            dst.write(patch[i], i+1)

# Assure que le dossier de sortie existe
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Filtrer les fichiers avec extension MS.tif ===
tif_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".tif")]

for i in range(NB_PATCHES):
    while True:
        tif_path = os.path.join(INPUT_FOLDER, random.choice(tif_files))
        with rasterio.open(tif_path) as src:
            if src.count >= 5: 
                break

    with rasterio.open(tif_path) as src:
        # Lire uniquement les 5 premiers canaux
        raster = src.read(indexes=[1, 2, 3, 4, 5])  # shape: (5, H, W)
        transform = src.transform
        crs = src.crs

        patch_5ch, x, y = get_random_valid_patch(raster)

        # Mise à l'échelle en 8-bit si besoin
        patch_scaled = (patch_5ch / patch_5ch.max() * 255).astype(np.uint8)

        # Transformer pour la zone crop
        window_transform = src.window_transform(((y, y + PATCH_SIZE), (x, x + PATCH_SIZE)))

        save_path = os.path.join(OUTPUT_FOLDER, f"sahel2021_{i}.tif")
        save_patch(save_path, patch_scaled, window_transform, crs)

        print(f"[{i}] Patch sauvegardé depuis {os.path.basename(tif_path)}")
