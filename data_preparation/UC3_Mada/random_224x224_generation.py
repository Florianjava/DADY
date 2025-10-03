import rasterio
from rasterio.mask import mask
import geopandas as gpd
import numpy as np
import random
import os


# === CONFIGURATION ===
OUTPUT_FOLDER = r"224x224_patchs"
FOLDER = r"data\UC3_riz"
SHP_PATH = r"data\UC3_riz\zones_safe.shp"
PATCH_SIZE = 224
MAX_TRIES = 1000
PIXEL_BLANC = 65535
PIXEL_NOIR = 0
NB_IMAGES = 25000

# === FONCTION POUR TESTER SI UN PATCH EST VALIDE ===
def is_valid_patch(x, y, valid_mask):
    mask_data = valid_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]

    # Vérifie que le patch est bien dans une zone valide
    if mask_data.shape != (PATCH_SIZE, PATCH_SIZE):
        return False
    if not mask_data.all():
        return False
    return True

# === CHERCHER UN PATCH ALÉATOIRE VALIDE ===
def get_random_patch(raster, width, height, valid_mask):
    for _ in range(MAX_TRIES):
        x = random.randint(0, width - PATCH_SIZE)
        y = random.randint(0, height - PATCH_SIZE)

        if is_valid_patch(x, y, valid_mask):
            patch = raster[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            return patch, x, y
    raise RuntimeError(f"Aucun patch valide trouvé après {MAX_TRIES} tentatives.")

        
def stack_channels(raster_path, x, y):
    """
    Stack all 5 channels (from channel_0 to channel_4) into a numpy array 
    for the given patch position (x, y).
    
    Returns:
        patch_stack: numpy array of shape (5, PATCH_SIZE, PATCH_SIZE)
    """
    patch_stack = []

    base_name = os.path.basename(raster_path)
    folder = os.path.dirname(raster_path)
    
    # Extract date and base info to reconstruct other paths
    parts = base_name.split("_")
    date_parts = "_".join(parts[:3])
    location = parts[3]
    
    for channel_num in range(5):
        channel_filename = f"{date_parts}_{location}_channel_{channel_num}.tif"
        channel_path = os.path.join(folder, channel_filename)

        with rasterio.open(channel_path) as src:
            band = src.read(1)
            patch = band[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_stack.append(patch)
    
    return np.stack(patch_stack)


def save_patch_tif(path, patch_stack, transform=None, crs=None):
    """
    Sauvegarde un patch 5 canaux en TIFF 8-bit.
    
    Args:
        path (str): chemin de sauvegarde (.tif)
        patch_stack (np.ndarray): tableau numpy shape (5, 224, 224), valeurs en uint8
        transform (Affine, optionnel): géotransformation à inclure dans le TIFF
        crs (CRS, optionnel): système de coordonnées
    
    Attention : patch_stack doit être de dtype uint8.
    """
    # if patch_stack.dtype != np.uint8:
        # patch_stack = patch_stack.astype(np.uint8)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=patch_stack.shape[1],
        width=patch_stack.shape[2],
        count=patch_stack.shape[0],
        dtype=rasterio.uint8,
        crs=crs,
        transform=transform
    ) as dst:
        for i in range(patch_stack.shape[0]):
            dst.write(patch_stack[i], i+1)

for i in range(NB_IMAGES):
    # === CHOISIR UN FICHIER tif ALÉATOIRE DANS LE DOSSIER ===
    random_tif = random.choice([f for f in os.listdir(FOLDER) if f.endswith(".tif")])
    random_tif_path = os.path.join(FOLDER, random_tif)

    # === LECTURE DU RASTER ET MASQUE DE ZONES PETES ===
    with rasterio.open(random_tif_path) as src:
        raster = src.read(1)  # Monochannel
        transform = src.transform
        crs = src.crs
        width, height = src.width, src.height
        valid_mask = raster

        # tif pété (2024_3_12_Andrano) : on exclut les zones du shapefile
        if random_tif.startswith("2024_3_12_Andrano"):
            # Charger le shapefile
            gdf = gpd.read_file(SHP_PATH)
            if gdf.crs != crs:
                gdf = gdf.to_crs(crs)

            # Appliquer le masque inverse pour obtenir les zones HORS shapefile
            out_image, _ = mask(src, gdf.geometry, invert=True, crop=False)
            valid_mask = out_image[0] 
        valid_mask = (valid_mask != PIXEL_BLANC) & (valid_mask != PIXEL_NOIR)

    # === TEST : EXTRAIRE, AFFICHER ET ANALYSER LES 5 CANAUX ===
    patch, x, y = get_random_patch(raster, width, height, valid_mask)
    window = ((y, y + PATCH_SIZE), (x, x + PATCH_SIZE))
    with rasterio.open(random_tif_path) as src:
        patch_transform = src.window_transform(window)

    patch_stack = stack_channels(random_tif_path, x, y)  # Shape: (5, 224, 224)
    # display_patch(patch_stack)
    patch_stack_uint8 = (patch_stack / patch_stack.max() * 255)
    save_patch_tif(f'{OUTPUT_FOLDER}/andrano{i}.tif', patch_stack_uint8, transform=patch_transform, crs=crs)
    print(random_tif_path)
