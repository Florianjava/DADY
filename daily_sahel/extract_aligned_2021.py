"""
extract_aligned_2021.py - Extraction des patches depuis les images recalées

Ce module extrait les patches pour chaque polygone du shapefile 9h,
en appliquant la transformation 9h→13h pour les extraire depuis
les images alignées (qui sont toutes dans l'espace 13h).

Workflow:
    1. Charger le shapefile 9h (ETM ou STR)
    2. Reprojeter dans le CRS du crop 9h
    3. Convertir les coordonnées géo → pixels dans le crop 9h
    4. Appliquer la transformation 9h→13h pour avoir les pixels dans l'espace aligné
    5. Extraire les patches depuis les images alignées
    6. Masquer l'extérieur du polygone (blanc/NaN)

Usage:
    python -m daily_sahel.extract_aligned_2021 --plot 1001 --hour 13h
    python -m daily_sahel.extract_aligned_2021 --all
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import mapping, box
from shapely.affinity import affine_transform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import warnings

# Supprimer les warnings rasterio pour fichiers sans géo-référence
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.nonlinear_registration import NonLinearTransform


# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoires
DATA_ROOT = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/anatomics"
CROPS_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/crops_2021"
ALIGNED_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/aligned_2021"
OUTPUT_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/patches_aligned_2021"

# Shapefile de référence (9h est le mieux annoté)
REFERENCE_HOUR = "9h"

# Heures disponibles
HOURS = ["09h", "11h", "13h", "15h", "17h"]

# Noms des bandes MS
MS_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6"]

# Marge autour du polygone (en pixels)
PATCH_MARGIN = 5

# Valeur pour masquer l'extérieur (NaN pour float, 0 pour int)
MASK_VALUE_FLOAT = np.nan
MASK_VALUE_INT = 0

# Coordonnées des moquettes (x, y) dans l'espace image aligné (13h)
MOQUETTES = {
    "Moquette_1": {"x": 3230, "y": 964},
    "Moquette_2": {"x": 3161, "y": 1326},
    "Moquette_3": {"x": 3111, "y": 1600},
}

# Taille du carré d'extraction pour les moquettes (en pixels)
MOQUETTE_PATCH_SIZE = 25

# Heure de référence pour la normalisation
NORMALIZATION_REFERENCE_HOUR = "09h"


# =============================================================================
# HELPERS
# =============================================================================

def plot_id_to_name(plot_id: int, half: str) -> str:
    """
    Convertit un plot_id en nom formaté.
    
    Ex: plot_id=1001, half="ETM" → "ETM_Rep1_001"
        plot_id=4128, half="STR" → "STR_Rep4_128"
    """
    rep = plot_id // 1000  # 1, 2, 3 ou 4
    num = plot_id % 1000   # 001 à 160
    return f"{half}_Rep{rep}_{num:03d}"


# =============================================================================
# NORMALISATION PAR LES MOQUETTES
# =============================================================================

def compute_moquette_reference(hour: str) -> Dict[str, float]:
    """
    Calcule la valeur moyenne des 3 moquettes pour chaque canal MS à une heure donnée.
    
    Args:
        hour: Heure ("09h", "11h", etc.)
    
    Returns:
        Dict avec la moyenne des 3 moquettes par bande
    """
    ms_path = os.path.join(ALIGNED_DIR, f"{hour}_MS_aligned.tif")
    
    if not os.path.exists(ms_path):
        print(f"⚠️ Fichier non trouvé: {ms_path}")
        return None
    
    # Accumuler les valeurs de chaque moquette
    band_values = {band: [] for band in MS_BAND_NAMES}
    
    with rasterio.open(ms_path) as src:
        ms_data = src.read()  # Shape: (n_bands, height, width)
        n_bands = ms_data.shape[0]
        
        for moquette_name, coords in MOQUETTES.items():
            x, y = coords["x"], coords["y"]
            half_size = MOQUETTE_PATCH_SIZE // 2
            
            # Extraire le patch centré sur la moquette
            y_min = max(0, y - half_size)
            y_max = min(ms_data.shape[1], y + half_size + 1)
            x_min = max(0, x - half_size)
            x_max = min(ms_data.shape[2], x + half_size + 1)
            
            patch = ms_data[:, y_min:y_max, x_min:x_max]
            
            # Calculer la moyenne par bande
            for band_idx in range(min(n_bands, len(MS_BAND_NAMES))):
                band_name = MS_BAND_NAMES[band_idx]
                band_patch = patch[band_idx]
                valid_values = band_patch[band_patch > 0]
                
                if len(valid_values) > 0:
                    band_values[band_name].append(float(np.mean(valid_values)))
    
    # Calculer la moyenne des 3 moquettes
    reference = {}
    for band_name in MS_BAND_NAMES:
        if band_values[band_name]:
            reference[band_name] = float(np.mean(band_values[band_name]))
        else:
            reference[band_name] = 1.0  # Éviter division par zéro
    
    return reference


def compute_all_normalization_coefficients() -> Dict[str, Dict[str, float]]:
    """
    Calcule les coefficients de normalisation pour toutes les heures.
    
    Le coefficient permet de ramener les valeurs à la référence 9h :
        coef[hour][band] = moquette_9h[band] / moquette_hour[band]
    
    Ainsi : valeur_normalisée = valeur_brute * coef
    
    Returns:
        Dict[hour][band] = coefficient multiplicateur
    """
    print("\n📊 Calcul des coefficients de normalisation (référence = 9h)...")
    
    # D'abord calculer la référence 9h
    ref_9h = compute_moquette_reference(NORMALIZATION_REFERENCE_HOUR)
    if ref_9h is None:
        print("❌ Impossible de calculer la référence 9h!")
        return {}
    
    print(f"   Référence 9h: " + ", ".join([f"{b}={v:.1f}" for b, v in ref_9h.items()]))
    
    coefficients = {}
    
    for hour in HOURS:
        ref_hour = compute_moquette_reference(hour)
        if ref_hour is None:
            continue
        
        coefs = {}
        for band_name in MS_BAND_NAMES:
            if ref_hour[band_name] > 0:
                coefs[band_name] = ref_9h[band_name] / ref_hour[band_name]
            else:
                coefs[band_name] = 1.0
        
        coefficients[hour] = coefs
        print(f"   {hour}: " + ", ".join([f"{b}={v:.3f}" for b, v in coefs.items()]))
    
    return coefficients


def name_to_plot_id(name: str) -> Tuple[int, str]:
    """
    Convertit un nom formaté en plot_id et half.
    
    Ex: "ETM_Rep1_001" → (1001, "ETM")
    """
    parts = name.split("_")
    half = parts[0]
    rep = int(parts[1].replace("Rep", ""))
    num = int(parts[2])
    plot_id = rep * 1000 + num
    return plot_id, half


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_shapefile_9h(half: str = "ETM") -> gpd.GeoDataFrame:
    """
    Charge le shapefile 9h (référence).
    
    Args:
        half: "ETM" ou "STR" (moitié du champ)
    
    Returns:
        GeoDataFrame avec les polygones
    """
    shp_path = os.path.join(DATA_ROOT, f"2021_{REFERENCE_HOUR}_{half}.shp")
    gdf = gpd.read_file(shp_path)
    print(f"📁 Chargé {shp_path}: {len(gdf)} polygones")
    return gdf


def load_transform_9h_to_13h() -> NonLinearTransform:
    """Charge la transformation 9h-NDVI → 13h-NDVI."""
    transform_path = os.path.join(ALIGNED_DIR, "transform_9h_ndvi_to_13h_ndvi")
    transform = NonLinearTransform.load(transform_path)
    print(f"📐 Transformation 9h→13h chargée")
    return transform


def get_crop_info(hour: str) -> Dict:
    """
    Récupère les infos du crop pour une heure donnée.
    
    Returns:
        Dict avec 'path', 'crs', 'transform', 'bounds', 'shape'
    """
    # Le crop NDVI existe pour toutes les heures
    crop_path = os.path.join(CROPS_DIR, f"{hour}_NDVI.tif")
    
    with rasterio.open(crop_path) as src:
        return {
            'path': crop_path,
            'crs': src.crs,
            'transform': src.transform,
            'bounds': src.bounds,
            'shape': (src.height, src.width),
            'profile': src.profile.copy()
        }


def get_aligned_paths(hour: str) -> Dict[str, str]:
    """Retourne les chemins des images alignées pour une heure."""
    return {
        'ms': os.path.join(ALIGNED_DIR, f"{hour}_MS_aligned.tif"),
        'ndvi': os.path.join(ALIGNED_DIR, f"{hour}_NDVI_aligned.tif"),
        'temp': os.path.join(ALIGNED_DIR, f"{hour}_Temp_aligned.tif"),
    }


# =============================================================================
# TRANSFORMATION DES COORDONNÉES
# =============================================================================

def geo_to_pixel(geom, transform) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convertit une géométrie en coordonnées pixels.
    
    Args:
        geom: Géométrie shapely (Polygon)
        transform: Transformation affine rasterio
    
    Returns:
        (cols, rows) : Arrays des coordonnées pixels des sommets
    """
    # Extraire les coordonnées du polygone
    if geom.geom_type == 'Polygon':
        coords = np.array(geom.exterior.coords)
    else:
        raise ValueError(f"Type de géométrie non supporté: {geom.geom_type}")
    
    xs, ys = coords[:, 0], coords[:, 1]
    
    # Conversion géo → pixel avec la transformation inverse
    inv_transform = ~transform
    cols, rows = inv_transform * (xs, ys)
    
    return cols.astype(np.float32), rows.astype(np.float32)


def transform_polygon_9h_to_13h(
    geom,
    crop_9h_info: Dict,
    crop_13h_info: Dict,
    transform: NonLinearTransform
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforme un polygone de l'espace 9h vers l'espace 13h.
    
    Stratégie :
        1. Convertir les coordonnées géo du shapefile 9h en pixels du crop 9h
        2. Appliquer la transformation 9h→13h (affine + non-linéaire)
        3. Retourner les coordonnées pixels dans l'espace 13h
    
    Args:
        geom: Géométrie du polygone (déjà reprojétée dans le CRS des crops)
        crop_9h_info: Infos du crop 9h
        crop_13h_info: Infos du crop 13h (images alignées)
        transform: Transformation 9h→13h (affine + non-linéaire)
    
    Returns:
        (cols_13h, rows_13h) : Coordonnées pixels dans l'espace 13h
    """
    # 1. Convertir géo → pixels dans le crop 9h
    cols_9h, rows_9h = geo_to_pixel(geom, crop_9h_info['transform'])
    
    # 2. Appliquer la transformation 9h→13h
    # transform_points attend un array (N,2) de coordonnées (x,y) = (col, row)
    pts_9h = np.column_stack([cols_9h, rows_9h])
    pts_13h = transform.transform_points(pts_9h)
    
    # 3. Extraire cols et rows dans l'espace 13h
    cols_13h = pts_13h[:, 0].astype(np.float32)
    rows_13h = pts_13h[:, 1].astype(np.float32)
    
    return cols_13h, rows_13h


# =============================================================================
# EXTRACTION DES PATCHES
# =============================================================================

def extract_patch_for_polygon(
    polygon_row: pd.Series,
    crop_9h_info: Dict,
    crop_13h_info: Dict,
    transform_9h_to_13h: NonLinearTransform,
    aligned_info: Dict,
    hour: str,
    half: str,
    norm_coefficients: Dict = None,
    output_dir: str = None,
    verbose: bool = True
) -> Dict:
    """
    Extrait le patch pour un polygone donné, depuis les images alignées.
    
    Args:
        polygon_row: Ligne du GeoDataFrame (avec geometry, Plot, etc.)
        crop_9h_info: Infos du crop 9h (pour référence)
        crop_13h_info: Infos du crop 13h (géo-référence des images alignées)
        transform_9h_to_13h: Transformation 9h→13h
        aligned_info: Infos de l'image alignée 13h (pour la taille)
        hour: Heure à extraire ("09h", "11h", etc.)
        half: "ETM" ou "STR"
        norm_coefficients: Coefficients de normalisation par heure et par bande (optionnel)
        output_dir: Répertoire de sortie
        verbose: Afficher les infos
    
    Returns:
        Dict avec les stats et chemins des fichiers créés
    """
    plot_id = polygon_row['Plot']
    plot_name = plot_id_to_name(plot_id, half)
    geom = polygon_row.geometry
    
    if verbose:
        print(f"\n   {plot_name} (plot {plot_id}):")
    
    # 1. Reprojeter la géométrie dans le CRS du crop 13h
    # (le shapefile est en EPSG:32628, le crop en EPSG:4326)
    gdf_temp = gpd.GeoDataFrame({'geometry': [geom]}, crs="EPSG:32628")
    gdf_temp = gdf_temp.to_crs(crop_13h_info['crs'])
    geom_reproj = gdf_temp.iloc[0].geometry
    
    # 2. Transformer le polygone vers l'espace 13h (images alignées)
    cols_13h, rows_13h = transform_polygon_9h_to_13h(
        geom_reproj, crop_9h_info, crop_13h_info, transform_9h_to_13h
    )
    
    # 3. Calculer la bounding box dans l'espace 13h
    col_min = int(np.floor(cols_13h.min())) - PATCH_MARGIN
    col_max = int(np.ceil(cols_13h.max())) + PATCH_MARGIN
    row_min = int(np.floor(rows_13h.min())) - PATCH_MARGIN
    row_max = int(np.ceil(rows_13h.max())) + PATCH_MARGIN
    
    # Clamp aux limites de l'image
    height_13h, width_13h = aligned_info['shape']
    col_min = max(0, col_min)
    col_max = min(width_13h, col_max)
    row_min = max(0, row_min)
    row_max = min(height_13h, row_max)
    
    patch_width = col_max - col_min
    patch_height = row_max - row_min
    
    if verbose:
        print(f"      BBox pixels 13h: [{col_min}, {row_min}] - [{col_max}, {row_max}]")
        print(f"      Taille patch: {patch_width} x {patch_height}")
    
    if patch_width <= 0 or patch_height <= 0:
        print(f"      ⚠️ Patch invalide!")
        return None
    
    # 4. Créer le masque du polygone dans l'espace du patch
    # On doit reconstruire la géométrie transformée
    from shapely.geometry import Polygon
    poly_13h = Polygon(zip(cols_13h, rows_13h))
    
    # Décaler pour être relatif au patch
    poly_patch = Polygon([(x - col_min, y - row_min) for x, y in zip(cols_13h, rows_13h)])
    
    mask = rasterize(
        [(mapping(poly_patch), 1)],
        out_shape=(patch_height, patch_width),
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    
    results = {
        'plot_name': plot_name,
        'plot_id': plot_id,
        'half': half,
        'hour': hour,
        'col_min': col_min,
        'row_min': row_min,
        'patch_width': patch_width,
        'patch_height': patch_height,
        'surface_pixels': int(np.sum(mask > 0))
    }
    
    # 5. Extraire depuis les images alignées
    aligned_paths = get_aligned_paths(hour)
    window = Window(col_min, row_min, patch_width, patch_height)
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_dir = os.path.join(output_dir, plot_name)
        os.makedirs(plot_dir, exist_ok=True)
        # Sous-dossiers par type
        ms_dir = os.path.join(plot_dir, "MS")
        ms_norm_dir = os.path.join(plot_dir, "MS_normalized")
        ndvi_dir = os.path.join(plot_dir, "NDVI")
        ndvi_norm_dir = os.path.join(plot_dir, "NDVI_normalized")
        temp_dir = os.path.join(plot_dir, "Temp")
        os.makedirs(ms_dir, exist_ok=True)
        os.makedirs(ms_norm_dir, exist_ok=True)
        os.makedirs(ndvi_dir, exist_ok=True)
        os.makedirs(ndvi_norm_dir, exist_ok=True)
        os.makedirs(temp_dir, exist_ok=True)
    
    # MS (original + normalisé)
    if os.path.exists(aligned_paths['ms']):
        with rasterio.open(aligned_paths['ms']) as src:
            ms_data = src.read(window=window)
            profile = src.profile.copy()
        
        # Masquer l'extérieur
        for b in range(ms_data.shape[0]):
            ms_data[b][mask == 0] = MASK_VALUE_INT
        
        if output_dir:
            # Sauvegarder MS original
            out_path = os.path.join(ms_dir, f"{plot_name}_{hour}_MS.tif")
            profile.update(width=patch_width, height=patch_height, count=ms_data.shape[0])
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(ms_data)
            results['ms_path'] = out_path
            if verbose:
                print(f"      ✅ MS: {out_path}")
            
            # Sauvegarder MS normalisé (si coefficients fournis)
            if norm_coefficients and hour in norm_coefficients:
                coefs = norm_coefficients[hour]
                ms_normalized = np.zeros_like(ms_data, dtype=np.float32)
                
                n_bands = ms_data.shape[0]
                for band_idx in range(min(n_bands, len(MS_BAND_NAMES))):
                    band_name = MS_BAND_NAMES[band_idx]
                    coef = coefs.get(band_name, 1.0)
                    ms_normalized[band_idx] = ms_data[band_idx].astype(np.float32) * coef
                    # Remettre le masque (les pixels masqués à 0 restent à 0 après multiplication)
                
                out_norm_path = os.path.join(ms_norm_dir, f"{plot_name}_{hour}_MS_normalized.tif")
                profile_norm = profile.copy()
                profile_norm.update(width=patch_width, height=patch_height, count=n_bands, dtype='float32')
                with rasterio.open(out_norm_path, 'w', **profile_norm) as dst:
                    dst.write(ms_normalized)
                results['ms_normalized_path'] = out_norm_path
                if verbose:
                    print(f"      ✅ MS_norm: {out_norm_path}")
                
                # Calculer et sauvegarder NDVI normalisé (à partir des MS normalisés)
                # NIR = bande 4 (index 4), Red = bande 2 (index 2)
                if n_bands >= 5:
                    nir_norm = ms_normalized[4]  # NIR normalisé
                    red_norm = ms_normalized[2]  # Red normalisé
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ndvi_normalized = (nir_norm - red_norm) / (nir_norm + red_norm)
                        ndvi_normalized = np.where(np.isfinite(ndvi_normalized), ndvi_normalized, -1)
                    # Masquer l'extérieur
                    ndvi_normalized[mask == 0] = MASK_VALUE_FLOAT
                    
                    out_ndvi_norm_path = os.path.join(ndvi_norm_dir, f"{plot_name}_{hour}_NDVI_normalized.tif")
                    with rasterio.open(out_ndvi_norm_path, 'w', driver='GTiff',
                                      width=patch_width, height=patch_height,
                                      count=1, dtype='float32') as dst:
                        dst.write(ndvi_normalized, 1)
                    results['ndvi_normalized_path'] = out_ndvi_norm_path
                    if verbose:
                        print(f"      ✅ NDVI_norm: {out_ndvi_norm_path}")
    
    # NDVI (original, depuis les images pré-calculées)
    if os.path.exists(aligned_paths['ndvi']):
        with rasterio.open(aligned_paths['ndvi']) as src:
            ndvi_data = src.read(1, window=window)
            profile = src.profile.copy()
        
        # Masquer l'extérieur
        ndvi_data[mask == 0] = MASK_VALUE_FLOAT
        
        if output_dir:
            out_path = os.path.join(ndvi_dir, f"{plot_name}_{hour}_NDVI.tif")
            profile.update(width=patch_width, height=patch_height, count=1)
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(ndvi_data, 1)
            results['ndvi_path'] = out_path
            if verbose:
                print(f"      ✅ NDVI: {out_path}")
        
        # Stats NDVI
        ndvi_valid = ndvi_data[mask > 0]
        if len(ndvi_valid) > 0:
            results['ndvi_mean'] = float(np.nanmean(ndvi_valid))
            results['ndvi_std'] = float(np.nanstd(ndvi_valid))
    
    # Temp
    if os.path.exists(aligned_paths['temp']):
        with rasterio.open(aligned_paths['temp']) as src:
            temp_data = src.read(1, window=window).astype(np.float32)  # Convertir en float
            profile = src.profile.copy()
        
        # Masquer l'extérieur
        temp_data[mask == 0] = MASK_VALUE_FLOAT
        
        if output_dir:
            out_path = os.path.join(temp_dir, f"{plot_name}_{hour}_Temp.tif")
            profile.update(width=patch_width, height=patch_height, count=1, dtype='float32')
            with rasterio.open(out_path, 'w', **profile) as dst:
                dst.write(temp_data, 1)
            results['temp_path'] = out_path
            if verbose:
                print(f"      ✅ Temp: {out_path}")
        
        # Stats Temp
        temp_valid = temp_data[mask > 0]
        if len(temp_valid) > 0:
            results['temp_mean'] = float(np.nanmean(temp_valid))
            results['temp_std'] = float(np.nanstd(temp_valid))
    
    # Sauvegarder le masque (une seule fois par plot, pas par heure)
    if output_dir:
        mask_path = os.path.join(plot_dir, f"{plot_name}_mask.tif")
        if not os.path.exists(mask_path):  # Ne pas réécrire si existe déjà
            with rasterio.open(mask_path, 'w', driver='GTiff', 
                              width=patch_width, height=patch_height, 
                              count=1, dtype=np.uint8) as dst:
                dst.write(mask, 1)
        results['mask_path'] = mask_path
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def extract_single_plot(plot_id: int, hour: str = "13h", half: str = None, verbose: bool = True):
    """
    Extrait les patches pour un seul plot, à une heure donnée.
    
    Args:
        plot_id: ID du plot (ex: 1001, 2042, etc.)
        hour: Heure à extraire
        half: "ETM" ou "STR" (auto-détecté si None)
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTION PLOT {plot_id} - {hour}")
    print(f"{'='*60}")
    
    # Déterminer la moitié (ETM ou STR) selon le plot_id
    if half is None:
        # Convention: R1-R2 (1xxx, 2xxx) = ETM, R3-R4 (3xxx, 4xxx) = STR
        thousands = plot_id // 1000
        half = "ETM" if thousands <= 2 else "STR"
    
    print(f"   Moitié: {half}")
    
    # Calculer les coefficients de normalisation
    norm_coefficients = compute_all_normalization_coefficients()
    
    # Charger les données
    gdf = load_shapefile_9h(half)
    transform = load_transform_9h_to_13h()
    crop_9h_info = get_crop_info("09h")
    crop_13h_info = get_crop_info("13h")  # Géo-référence des images alignées
    aligned_info = crop_13h_info  # Même info
    
    # Trouver le polygone
    polygon_row = gdf[gdf['Plot'] == plot_id]
    if len(polygon_row) == 0:
        print(f"❌ Plot {plot_id} non trouvé dans {half}")
        return None
    
    polygon_row = polygon_row.iloc[0]
    
    # Extraire
    result = extract_patch_for_polygon(
        polygon_row,
        crop_9h_info,
        crop_13h_info,
        transform,
        aligned_info,
        hour,
        half,
        norm_coefficients=norm_coefficients,
        output_dir=OUTPUT_DIR,
        verbose=verbose
    )
    
    return result


def extract_all_plots(hours: List[str] = None, half: str = "ETM", max_plots: int = None):
    """
    Extrait les patches pour tous les plots.
    
    Args:
        hours: Liste des heures à traiter (défaut: toutes)
        half: "ETM" ou "STR" 
        max_plots: Limite le nombre de plots (pour test)
    """
    if hours is None:
        hours = HOURS
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION DE TOUS LES PLOTS - {half}")
    print(f"Heures: {hours}")
    print(f"{'='*60}")
    
    # Calculer les coefficients de normalisation (une seule fois pour tous)
    norm_coefficients = compute_all_normalization_coefficients()
    
    # Charger les données communes
    gdf = load_shapefile_9h(half)
    transform = load_transform_9h_to_13h()
    crop_9h_info = get_crop_info("09h")
    crop_13h_info = get_crop_info("13h")  # Géo-référence des images alignées
    aligned_info = crop_13h_info
    
    if max_plots:
        gdf = gdf.head(max_plots)
        print(f"⚠️ Limité à {max_plots} plots (test)")
    
    all_results = []
    
    for hour in hours:
        print(f"\n{'#'*60}")
        print(f"# HEURE: {hour}")
        print(f"{'#'*60}")
        
        t0 = time.time()
        
        for idx, polygon_row in gdf.iterrows():
            result = extract_patch_for_polygon(
                polygon_row,
                crop_9h_info,
                crop_13h_info,
                transform,
                aligned_info,
                hour,
                half,
                norm_coefficients=norm_coefficients,
                output_dir=OUTPUT_DIR,
                verbose=False
            )
            
            if result:
                all_results.append(result)
            
            # Progress
            if (idx + 1) % 50 == 0:
                print(f"   {idx + 1}/{len(gdf)} plots traités...")
        
        elapsed = time.time() - t0
        print(f"   ✅ {len(gdf)} plots en {elapsed:.1f}s")
    
    # Sauvegarder le résumé
    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUTPUT_DIR, f"extraction_summary_{half}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n📄 Résumé sauvegardé: {csv_path}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Extraction des patches depuis les images recalées"
    )
    parser.add_argument("--plot", type=int, help="ID du plot à extraire")
    parser.add_argument("--hour", type=str, default="13h", help="Heure à extraire")
    parser.add_argument("--half", type=str, choices=["ETM", "STR"], help="ETM ou STR (défaut: les deux avec --all)")
    parser.add_argument("--all", action="store_true", help="Extraire tous les plots (ETM + STR)")
    parser.add_argument("--max-plots", type=int, help="Limite le nombre de plots par shapefile (test)")
    parser.add_argument("--hours", type=str, help="Heures à traiter (ex: 09h,13h,17h)")
    
    args = parser.parse_args()
    
    if args.plot:
        extract_single_plot(args.plot, args.hour, args.half)
    elif args.all:
        hours = args.hours.split(",") if args.hours else None
        
        # Si --half est spécifié, extraire seulement celui-là
        # Sinon, extraire ETM et STR
        halves = [args.half] if args.half else ["ETM", "STR"]
        
        for half in halves:
            extract_all_plots(hours=hours, half=half, max_plots=args.max_plots)
        
        print(f"\n{'='*60}")
        print(f"✅ EXTRACTION TERMINÉE")
        print(f"   Shapefiles: {halves}")
        print(f"   Total dossiers: {640 * len(halves)}")
        print(f"{'='*60}")
    else:
        parser.print_help()
        print("\nExemples:")
        print("  python -m daily_sahel.extract_aligned_2021 --plot 1001 --hour 13h --half ETM")
        print("  python -m daily_sahel.extract_aligned_2021 --all  # ETM + STR")
        print("  python -m daily_sahel.extract_aligned_2021 --all --half ETM  # ETM seulement")
        print("  python -m daily_sahel.extract_aligned_2021 --all --max-plots 10  # Test")


if __name__ == "__main__":
    main()
