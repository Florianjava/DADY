#!/usr/bin/env python3
"""
prepare_rgb.py - Préparation des images RGB haute résolution

Ce script prépare les images RGB (ETM et STR) pour le recalage :
    1. Détection automatique des bords de la zone utile (fond blanc = 255,255,255)
    2. Conversion RGB → niveau de gris (moyenne des 3 canaux)
    3. Crop selon les coordonnées détectées ou fournies
    4. Préparation pour annotation des correspondances

Les fichiers originaux sont :
    - 2022_14h_ETM_RGB.tif : moitié ETM à 14h
    - 2022_14h_STR_RGB.tif : moitié STR à 14h
    - 2022_10h_ETM_RGB.tif : moitié ETM à 10h
    - 2022_10h_STR_RGB.tif : moitié STR à 10h

Résolution : ~0.9mm/pixel (32230 x 52280 pixels)

Usage:
    python prepare_rgb.py --info                    # Affiche les infos des fichiers
    python prepare_rgb.py --detect                  # Détecte les bords et crée le CSV
    python prepare_rgb.py --convert                 # Convertit RGB → grayscale
    python prepare_rgb.py --crop                    # Crop selon coordonnées CSV
    python prepare_rgb.py --all                     # Fait tout (detect + convert + crop)
"""

import os
import sys
import argparse
import numpy as np
import rasterio
from rasterio.windows import Window
import pandas as pd
from typing import Tuple, Dict, Optional
import time

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/anatomics"
WORK_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"
OUTPUT_DIR = os.path.join(WORK_DIR, "crops_2021")  # Même dossier que les autres crops

# Fichiers RGB disponibles (seulement 14h pour le moment)
RGB_FILES = {
    "14h_ETM": "2022_14h_ETM_RGB.tif",
    "14h_STR": "2022_14h_STR_RGB.tif",
    # "10h_ETM": "2022_10h_ETM_RGB.tif",  # TODO: à ajouter plus tard
    # "10h_STR": "2022_10h_STR_RGB.tif",  # TODO: à ajouter plus tard
}

# Fichier CSV pour les coordonnées de crop RGB
CROP_COORDS_CSV = os.path.join(WORK_DIR, "crop_coordinates_rgb.csv")

# Valeur du fond (blanc)
BACKGROUND_VALUE = 255

# Marge à ajouter autour de la zone détectée (en pixels)
MARGIN = 10


# =============================================================================
# FONCTIONS D'INFORMATION
# =============================================================================

def get_rgb_info(name: str) -> Dict:
    """Récupère les informations d'un fichier RGB."""
    filepath = os.path.join(DATA_DIR, RGB_FILES[name])
    
    if not os.path.exists(filepath):
        return {"error": f"Fichier non trouvé: {filepath}"}
    
    with rasterio.open(filepath) as src:
        return {
            "name": name,
            "filepath": filepath,
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "transform": src.transform,
            "bounds": src.bounds,
            "pixel_size": (src.transform[0], -src.transform[4]),
            "size_mb": os.path.getsize(filepath) / (1024 * 1024)
        }


def print_all_rgb_info():
    """Affiche les informations de tous les fichiers RGB."""
    print("\n" + "=" * 70)
    print("INFORMATIONS DES FICHIERS RGB")
    print("=" * 70)
    
    for name in RGB_FILES:
        info = get_rgb_info(name)
        
        if "error" in info:
            print(f"\n❌ {name}: {info['error']}")
            continue
        
        print(f"\n📁 {name}")
        print(f"   Fichier: {info['filepath']}")
        print(f"   Taille: {info['width']} x {info['height']} pixels")
        print(f"   Bandes: {info['bands']} ({info['dtype']})")
        print(f"   Résolution: {info['pixel_size'][0]*1000:.2f} mm/pixel")
        print(f"   Taille fichier: {info['size_mb']:.0f} MB")


# =============================================================================
# DÉTECTION AUTOMATIQUE DES BORDS
# =============================================================================

def detect_bounds(name: str, sample_step: int = 500) -> Dict:
    """
    Détecte automatiquement les bords de la zone utile dans une image RGB.
    
    Le fond est supposé être blanc (255, 255, 255).
    On scanne l'image par lignes/colonnes échantillonnées pour trouver
    xmin, xmax, ymin, ymax de la zone non-blanche.
    
    Args:
        name: Nom du fichier (ex: "14h_ETM")
        sample_step: Pas d'échantillonnage pour accélérer (en pixels)
    
    Returns:
        Dict avec xmin, xmax, ymin, ymax, width, height
    """
    filepath = os.path.join(DATA_DIR, RGB_FILES[name])
    
    print(f"\n🔍 Détection des bords pour {name}...")
    t0 = time.time()
    
    with rasterio.open(filepath) as src:
        height, width = src.height, src.width
        
        # On va scanner par bandes horizontales et verticales
        # pour trouver les limites de la zone utile
        
        xmin, xmax = width, 0
        ymin, ymax = height, 0
        
        # Scanner des lignes horizontales échantillonnées
        n_lines = height // sample_step
        print(f"   Scan horizontal ({n_lines} lignes)...")
        for i, y in enumerate(range(0, height, sample_step)):
            if i % 20 == 0:
                print(f"      Ligne {i}/{n_lines} (y={y})...", end="\r")
            # Lire une ligne complète (tous les canaux)
            row_data = src.read(window=Window(0, y, width, 1))  # Shape: (3, 1, width)
            
            # Chercher les pixels non-blancs (au moins un canal != 255)
            non_white = np.any(row_data != BACKGROUND_VALUE, axis=0).flatten()
            
            if np.any(non_white):
                indices = np.where(non_white)[0]
                xmin = min(xmin, indices[0])
                xmax = max(xmax, indices[-1])
                ymin = min(ymin, y)
                ymax = max(ymax, y)
        
        # Scanner des colonnes verticales échantillonnées pour affiner ymin/ymax
        n_cols = (xmax - xmin + 1) // sample_step
        print(f"\n   Scan vertical ({n_cols} colonnes)...")
        for i, x in enumerate(range(xmin, xmax + 1, sample_step)):
            if i % 20 == 0:
                print(f"      Colonne {i}/{n_cols} (x={x})...", end="\r")
            # Lire une colonne complète
            col_data = src.read(window=Window(x, 0, 1, height))  # Shape: (3, height, 1)
            
            # Chercher les pixels non-blancs
            non_white = np.any(col_data != BACKGROUND_VALUE, axis=0).flatten()
            
            if np.any(non_white):
                indices = np.where(non_white)[0]
                ymin = min(ymin, indices[0])
                ymax = max(ymax, indices[-1])
        
        # Affiner les bords avec un scan plus précis près des limites
        print(f"\n   Affinage des bords...")
        
        # Affiner xmin
        print(f"      Affinage xmin (de {max(0, xmin - sample_step)} à {xmin})...")
        for x in range(max(0, xmin - sample_step), xmin + 1):
            col_data = src.read(window=Window(x, ymin, 1, ymax - ymin + 1))
            if np.any(col_data != BACKGROUND_VALUE):
                xmin = x
                break
        
        # Affiner xmax
        print(f"      Affinage xmax (de {min(width - 1, xmax + sample_step)} à {xmax})...")
        for x in range(min(width - 1, xmax + sample_step), xmax - 1, -1):
            col_data = src.read(window=Window(x, ymin, 1, ymax - ymin + 1))
            if np.any(col_data != BACKGROUND_VALUE):
                xmax = x
                break
        
        # Affiner ymin
        print(f"      Affinage ymin (de {max(0, ymin - sample_step)} à {ymin})...")
        for y in range(max(0, ymin - sample_step), ymin + 1):
            row_data = src.read(window=Window(xmin, y, xmax - xmin + 1, 1))
            if np.any(row_data != BACKGROUND_VALUE):
                ymin = y
                break
        
        # Affiner ymax
        print(f"      Affinage ymax (de {min(height - 1, ymax + sample_step)} à {ymax})...")
        for y in range(min(height - 1, ymax + sample_step), ymax - 1, -1):
            row_data = src.read(window=Window(xmin, y, xmax - xmin + 1, 1))
            if np.any(row_data != BACKGROUND_VALUE):
                ymax = y
                break
        
        # Ajouter une marge
        xmin = max(0, xmin - MARGIN)
        ymin = max(0, ymin - MARGIN)
        xmax = min(width - 1, xmax + MARGIN)
        ymax = min(height - 1, ymax + MARGIN)
        
        crop_width = xmax - xmin + 1
        crop_height = ymax - ymin + 1
        
        elapsed = time.time() - t0
        print(f"   ✅ Détection terminée en {elapsed:.1f}s")
        print(f"   Zone utile: x=[{xmin}, {xmax}], y=[{ymin}, {ymax}]")
        print(f"   Taille crop: {crop_width} x {crop_height} pixels")
        
        return {
            "name": name,
            "xoff": xmin,
            "yoff": ymin,
            "width": crop_width,
            "height": crop_height,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
            "original_width": width,
            "original_height": height
        }


def detect_all_bounds_and_save():
    """Détecte les bords de tous les fichiers RGB et sauvegarde dans le CSV."""
    results = []
    
    print("\n" + "=" * 70)
    print("DÉTECTION AUTOMATIQUE DES BORDS")
    print("=" * 70)
    
    for name in RGB_FILES:
        try:
            bounds = detect_bounds(name)
            results.append({
                "name": bounds["name"],
                "xoff": bounds["xoff"],
                "yoff": bounds["yoff"],
                "width": bounds["width"],
                "height": bounds["height"]
            })
        except Exception as e:
            print(f"❌ Erreur pour {name}: {e}")
    
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(CROP_COORDS_CSV), exist_ok=True)
        df.to_csv(CROP_COORDS_CSV, index=False)
        print(f"\n📝 Coordonnées sauvegardées: {CROP_COORDS_CSV}")
        print(df.to_string(index=False))
    
    return results


# =============================================================================
# CONVERSION RGB → GRAYSCALE
# =============================================================================

def convert_rgb_to_grayscale(
    name: str,
    output_dir: str = None,
    block_size: int = 1024
) -> str:
    """
    Convertit un fichier RGB en niveau de gris (moyenne des 3 canaux).
    
    Traitement par blocs pour gérer les gros fichiers.
    
    Args:
        name: Nom du fichier (ex: "14h_ETM")
        output_dir: Dossier de sortie (si None, même dossier que l'original)
        block_size: Taille des blocs pour le traitement
    
    Returns:
        Chemin du fichier grayscale créé
    """
    src_path = os.path.join(DATA_DIR, RGB_FILES[name])
    
    if output_dir is None:
        output_dir = DATA_DIR
    
    dst_path = os.path.join(output_dir, f"{name}_gray.tif")
    
    print(f"\n🔄 Conversion {name} RGB → Grayscale")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    t0 = time.time()
    
    with rasterio.open(src_path) as src:
        print(f"   Taille image: {src.width} x {src.height} pixels")
        
        # Créer le profil de sortie (1 seule bande, float32 pour la moyenne)
        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype='float32',
            compress='lzw',
            tiled=True,
            blockxsize=256,
            blockysize=256
        )
        
        height, width = src.height, src.width
        
        print(f"   Création du fichier de sortie...")
        
        with rasterio.open(dst_path, 'w', **profile) as dst:
            # Traitement par blocs
            n_blocks_y = (height + block_size - 1) // block_size
            n_blocks_x = (width + block_size - 1) // block_size
            total_blocks = n_blocks_y * n_blocks_x
            
            print(f"   Traitement par blocs: {n_blocks_x} x {n_blocks_y} = {total_blocks} blocs de {block_size}x{block_size}")
            
            block_count = 0
            last_print = time.time()
            for row_idx, row in enumerate(range(0, height, block_size)):
                for col in range(0, width, block_size):
                    block_count += 1
                    
                    # Taille du bloc (peut être plus petit aux bords)
                    h = min(block_size, height - row)
                    w = min(block_size, width - col)
                    
                    window = Window(col, row, w, h)
                    
                    # Lire les 3 bandes
                    rgb = src.read(window=window)  # Shape: (3, h, w)
                    
                    # Moyenne des canaux
                    gray = np.mean(rgb.astype(np.float32), axis=0)
                    
                    # Écrire
                    dst.write(gray, 1, window=window)
                
                # Print à chaque ligne de blocs
                elapsed = time.time() - t0
                speed = block_count / elapsed if elapsed > 0 else 0
                remaining = (total_blocks - block_count) / speed if speed > 0 else 0
                print(f"   Ligne {row_idx+1}/{n_blocks_y} ({100*block_count/total_blocks:.1f}%) - {elapsed:.0f}s écoulé, ~{remaining:.0f}s restant")
    
    elapsed = time.time() - t0
    print(f"✅ Fichier créé en {elapsed:.1f}s: {dst_path}")
    return dst_path


def convert_all_rgb_to_grayscale():
    """Convertit tous les fichiers RGB en grayscale."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for name in RGB_FILES:
        try:
            convert_rgb_to_grayscale(name, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Erreur pour {name}: {e}")


# =============================================================================
# CROP DES IMAGES
# =============================================================================

def load_crop_coordinates_rgb() -> pd.DataFrame:
    """Charge les coordonnées de crop depuis le CSV."""
    if not os.path.exists(CROP_COORDS_CSV):
        print(f"⚠️  Fichier de coordonnées non trouvé: {CROP_COORDS_CSV}")
        print("   Créez ce fichier avec les colonnes: name,xoff,yoff,width,height")
        return None
    
    return pd.read_csv(CROP_COORDS_CSV)


def create_crop_template():
    """Crée un template CSV pour les coordonnées de crop."""
    template = pd.DataFrame({
        'name': ['14h_ETM', '14h_STR', '10h_ETM', '10h_STR'],
        'xoff': [0, 0, 0, 0],
        'yoff': [0, 0, 0, 0],
        'width': [5700, 5700, 5700, 5700],
        'height': [6800, 6800, 6800, 6800]
    })
    
    os.makedirs(os.path.dirname(CROP_COORDS_CSV), exist_ok=True)
    template.to_csv(CROP_COORDS_CSV, index=False)
    print(f"📝 Template créé: {CROP_COORDS_CSV}")
    print("   Éditez ce fichier avec les bonnes coordonnées de crop")
    return CROP_COORDS_CSV


def crop_rgb_grayscale(
    name: str,
    xoff: int,
    yoff: int,
    width: int,
    height: int,
    output_dir: str = None
) -> str:
    """
    Crop une image grayscale RGB.
    
    Args:
        name: Nom du fichier (ex: "14h_ETM")
        xoff, yoff: Coin supérieur gauche du crop
        width, height: Taille du crop
        output_dir: Dossier de sortie
    
    Returns:
        Chemin du fichier croppé
    """
    # Chercher d'abord le fichier grayscale, sinon l'original
    gray_path = os.path.join(OUTPUT_DIR, f"{name}_gray.tif")
    if not os.path.exists(gray_path):
        print(f"⚠️  Fichier grayscale non trouvé: {gray_path}")
        print("   Lancez d'abord --convert pour créer les fichiers grayscale")
        return None
    
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    dst_path = os.path.join(output_dir, f"{name}_gray_crop.tif")
    
    print(f"\n✂️  Crop {name}")
    print(f"   Source: {gray_path}")
    print(f"   Zone: ({xoff}, {yoff}) - taille {width}x{height}")
    
    t0 = time.time()
    
    with rasterio.open(gray_path) as src:
        print(f"   Taille image source: {src.width} x {src.height} pixels")
        
        # Vérifier que le crop est dans les limites
        if xoff + width > src.width or yoff + height > src.height:
            print(f"❌ Crop hors limites! Image: {src.width}x{src.height}")
            return None
        
        print(f"   Lecture de la zone...")
        window = Window(xoff, yoff, width, height)
        data = src.read(window=window)
        print(f"   Zone lue: {data.shape}")
        
        # Calculer la nouvelle transformation
        transform = src.window_transform(window)
        
        profile = src.profile.copy()
        profile.update(
            width=width,
            height=height,
            transform=transform
        )
        
        print(f"   Écriture du fichier crop...")
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)
    
    elapsed = time.time() - t0
    print(f"✅ Fichier créé en {elapsed:.1f}s: {dst_path}")
    return dst_path


def crop_all_rgb():
    """Crop tous les fichiers RGB selon les coordonnées du CSV."""
    coords = load_crop_coordinates_rgb()
    
    if coords is None:
        create_crop_template()
        return
    
    for _, row in coords.iterrows():
        try:
            crop_rgb_grayscale(
                row['name'],
                int(row['xoff']),
                int(row['yoff']),
                int(row['width']),
                int(row['height'])
            )
        except Exception as e:
            print(f"❌ Erreur pour {row['name']}: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Préparation des images RGB haute résolution"
    )
    parser.add_argument("--info", action="store_true", help="Affiche les infos des fichiers")
    parser.add_argument("--detect", action="store_true", help="Détecte les bords et crée le CSV")
    parser.add_argument("--convert", action="store_true", help="Convertit RGB → grayscale")
    parser.add_argument("--crop", action="store_true", help="Crop selon coordonnées CSV")
    parser.add_argument("--template", action="store_true", help="Crée le template CSV de coordonnées")
    parser.add_argument("--all", action="store_true", help="Fait tout (detect + convert + crop)")
    
    args = parser.parse_args()
    
    if args.info or (not any([args.detect, args.convert, args.crop, args.template, args.all])):
        print_all_rgb_info()
    
    if args.template:
        create_crop_template()
    
    if args.detect or args.all:
        detect_all_bounds_and_save()
    
    if args.convert or args.all:
        convert_all_rgb_to_grayscale()
    
    if args.crop or args.all:
        crop_all_rgb()


if __name__ == "__main__":
    main()
