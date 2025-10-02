import rasterio
import numpy as np
import os
import random

# === CONFIGURATION ===
INPUT_FOLDER = r"C:\Users\fdubois\Desktop\data_igname"
OUTPUT_FOLDER = r"224x224_patchs"
PATCH_SIZE = 224
NB_PATCHES = 10000
MAX_TRIES = 1000
PIXEL_NOIR = 0
PIXEL_BLANC = 65535

def is_valid_patch(patch):
    """
    Vérifie si le patch ne contient ni pixels noirs ni blancs dans aucun canal.
    """
    return not ((patch == PIXEL_NOIR).any() or (patch == PIXEL_BLANC).any())

def get_random_valid_patch(raster):
    """
    Essaie de trouver un patch valide dans le raster.
    """
    _, height, width = raster.shape
    for _ in range(MAX_TRIES):
        x = random.randint(0, width - PATCH_SIZE)
        y = random.randint(0, height - PATCH_SIZE)
        patch = raster[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        if patch.shape[1:] == (PATCH_SIZE, PATCH_SIZE) and is_valid_patch(patch):
            return patch, x, y
    raise RuntimeError("Aucun patch valide trouvé.")


def save_patch(path, patch, transform, crs):
    """
    Sauvegarde un patch 5 canaux en TIFF 8-bit.
    """
    patch = patch.astype(np.uint8)
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=PATCH_SIZE,
        width=PATCH_SIZE,
        count=5,  # 5 canaux maintenant
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(5):
            dst.write(patch[i], i+1)

# === FILTRER LES FICHIERS FINISSANT PAR 'MS.tif' (rasters multispectraux) ===
tif_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith("MS.tif")]

for i in range(NB_PATCHES):
    # Choisir un fichier avec exactement 4 canaux
    while True:
        tif_path = os.path.join(INPUT_FOLDER, random.choice(tif_files))
        with rasterio.open(tif_path) as src:
            if src.count == 4:
                break  # on garde ce fichier

    with rasterio.open(tif_path) as src:
        raster = src.read()  # shape: (4, H, W)
        transform = src.transform
        crs = src.crs

        patch_4ch, x, y = get_random_valid_patch(raster)

        # Créer un canal de zéros et l'ajouter en tête
        zero_channel = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=patch_4ch.dtype)
        patch_5ch = np.vstack(([zero_channel], patch_4ch))  # shape: (5, 224, 224)

        # Mise à l'échelle sur 8 bits
        patch_scaled = (patch_5ch / patch_5ch.max() * 255).astype(np.uint8)

        # Calcul du transform du patch
        window_transform = src.window_transform(((y, y + PATCH_SIZE), (x, x + PATCH_SIZE)))

        save_path = os.path.join(OUTPUT_FOLDER, f"GodetC1_{i}.tif")
        save_patch(save_path, patch_scaled, window_transform, crs)

        print(f"[{i}] Patch sauvegardé depuis {os.path.basename(tif_path)}")
