#!/usr/bin/env python3
"""
apply_all_transforms.py - Applique les transformations à toutes les images

Ce script recalcule les transformations à la volée à partir des CSV de points
de correspondance, puis les applique aux images MS et Temp.

Avantage : pas besoin de stocker les champs de déformation (600 Mo chacun).

Workflow des transformations :
    - MS/NDVI Xh → 13h-NDVI : transformation directe (Xh-ndvi_to_13-ndvi)
    - Temp Xh → 13h-NDVI : composition de 2 transformations :
        1. Temp Xh → 13h-Temp (Xh-temp_to_13-temp)
        2. 13h-Temp → 13h-NDVI (13-temp_to_13-ndvi)

Usage:
    python apply_all_transforms.py [--ms] [--temp] [--ndvi] [--hour 09h]
    python apply_all_transforms.py --all   # Tout faire
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
import time

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.nonlinear_registration import NonLinearTransform, compose_transforms

# =============================================================================
# CONFIGURATION
# =============================================================================

CROPS_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/crops_2021"
ALIGNED_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/aligned_2021"
ROIS_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/Rois_0_align_crop_affine"

# Heures à traiter (13h est la référence)
HOURS = ["09h", "11h", "15h", "17h"]
REF_HOUR = "13h"

# Taille de sortie (13h-NDVI)
OUTPUT_SHAPE = (6800, 5700)  # (height, width)


# =============================================================================
# CHARGEMENT DES POINTS DE CORRESPONDANCE
# =============================================================================

def load_points_from_csv(csv_pattern: str) -> tuple:
    """
    Charge et fusionne les points de correspondance depuis 2 CSV (set1 + set2).
    
    Args:
        csv_pattern: Pattern du CSV (ex: "RoiSet_9-ndvi_to_13-ndvi")
    
    Returns:
        (pts_src, pts_ref) : arrays (N, 2) de coordonnées (x, y)
    """
    csv1 = os.path.join(ROIS_DIR, f"{csv_pattern}_set1.csv")
    csv2 = os.path.join(ROIS_DIR, f"{csv_pattern}_set2.csv")
    
    if not os.path.exists(csv1):
        raise FileNotFoundError(f"CSV non trouvé: {csv1}")
    
    # Charger set1 (source) et set2 (référence)
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    
    # Extraire les coordonnées (colonnes en minuscules)
    pts_src = df1[['x', 'y']].values
    pts_ref = df2[['x', 'y']].values
    
    print(f"      Chargé {len(pts_src)} points depuis {csv_pattern}")
    
    return pts_src, pts_ref


def compute_transform_from_csv(csv_pattern: str, output_shape: tuple) -> NonLinearTransform:
    """
    Calcule une transformation non-linéaire à partir des CSV de points.
    
    Args:
        csv_pattern: Pattern du CSV (ex: "RoiSet_9-ndvi_to_13-ndvi")
        output_shape: Taille de sortie (height, width)
    
    Returns:
        NonLinearTransform calculé
    """
    pts_src, pts_ref = load_points_from_csv(csv_pattern)
    
    transform = NonLinearTransform()
    transform.compute_from_points(
        pts_ref=pts_ref,
        pts_src=pts_src,
        output_shape=output_shape,
        mode='affine+nonlinear',
        sigma=0  # Auto
    )
    
    return transform


# =============================================================================
# CALCUL DES TRANSFORMATIONS PAR TYPE
# =============================================================================

def get_ndvi_transform(hour: str) -> NonLinearTransform:
    """
    Calcule la transformation NDVI Xh → 13h-NDVI.
    
    Args:
        hour: Heure (ex: "09h", "11h", etc.)
    
    Returns:
        NonLinearTransform
    """
    hour_num = hour.replace("h", "").lstrip("0") or "0"
    csv_pattern = f"RoiSet_{hour_num}-ndvi_to_13-ndvi"
    
    print(f"   📐 Calcul transformation NDVI {hour} → 13h-NDVI")
    return compute_transform_from_csv(csv_pattern, OUTPUT_SHAPE)


def get_temp_transform(hour: str) -> NonLinearTransform:
    """
    Calcule la transformation composée Temp Xh → 13h-NDVI.
    
    Composition :
        1. Temp Xh → 13h-Temp
        2. 13h-Temp → 13h-NDVI
    
    Args:
        hour: Heure (ex: "09h", "11h", etc.)
    
    Returns:
        NonLinearTransform composé
    """
    hour_num = hour.replace("h", "").lstrip("0") or "0"
    
    print(f"   📐 Calcul transformation Temp {hour} → 13h-NDVI (composée)")
    
    # Étape 1 : Temp Xh → 13h-Temp
    # On a besoin de la taille intermédiaire (13h-Temp)
    temp_13h_path = os.path.join(CROPS_DIR, "13h_Temp_crop.tif")
    with rasterio.open(temp_13h_path) as src:
        intermediate_shape = (src.height, src.width)
    
    print(f"      Étape 1: Temp {hour} → 13h-Temp (taille intermédiaire: {intermediate_shape[1]}x{intermediate_shape[0]})")
    csv_pattern1 = f"RoiSet_{hour_num}-temp_to_13-temp"
    t1 = compute_transform_from_csv(csv_pattern1, intermediate_shape)
    
    # Étape 2 : 13h-Temp → 13h-NDVI
    print(f"      Étape 2: 13h-Temp → 13h-NDVI")
    csv_pattern2 = "RoiSet_13-temp_to_13-ndvi"
    t2 = compute_transform_from_csv(csv_pattern2, OUTPUT_SHAPE)
    
    # Composition
    print(f"      Composition des transformations...")
    composed = compose_transforms(t1, t2)
    
    return composed


def get_temp_transform_13h() -> NonLinearTransform:
    """
    Calcule la transformation 13h-Temp → 13h-NDVI (cas spécial).
    
    Returns:
        NonLinearTransform
    """
    print(f"   📐 Calcul transformation 13h-Temp → 13h-NDVI")
    csv_pattern = "RoiSet_13-temp_to_13-ndvi"
    return compute_transform_from_csv(csv_pattern, OUTPUT_SHAPE)


# =============================================================================
# APPLICATION DES TRANSFORMATIONS
# =============================================================================

def apply_transform(src_path: str, dst_path: str, transform: NonLinearTransform) -> bool:
    """
    Applique une transformation à une image.
    
    Args:
        src_path: Chemin source
        dst_path: Chemin destination
        transform: Transformation à appliquer
    
    Returns:
        True si succès
    """
    # Charger le profil de référence pour le géoréférencement
    ref_path = os.path.join(CROPS_DIR, "13h_NDVI.tif")
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
    
    return transform.apply_to_file(src_path, dst_path, ref_profile=ref_profile)


def process_ms(hour: str) -> bool:
    """Traite une image MS."""
    src_path = os.path.join(CROPS_DIR, f"{hour}_MS_crop.tif")
    dst_path = os.path.join(ALIGNED_DIR, f"{hour}_MS_aligned.tif")
    
    print(f"\n🔄 MS {hour}")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    t0 = time.time()
    transform = get_ndvi_transform(hour)
    
    print(f"   Application...")
    success = apply_transform(src_path, dst_path, transform)
    
    elapsed = time.time() - t0
    if success:
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"   ✅ OK en {elapsed:.1f}s ({size_mb:.0f} MB)")
    else:
        print(f"   ❌ Échec après {elapsed:.1f}s")
    
    return success


def process_ndvi(hour: str) -> bool:
    """Traite une image NDVI."""
    src_path = os.path.join(CROPS_DIR, f"{hour}_NDVI.tif")
    dst_path = os.path.join(ALIGNED_DIR, f"{hour}_NDVI_aligned.tif")
    
    print(f"\n🔄 NDVI {hour}")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    t0 = time.time()
    transform = get_ndvi_transform(hour)
    
    print(f"   Application...")
    success = apply_transform(src_path, dst_path, transform)
    
    elapsed = time.time() - t0
    if success:
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"   ✅ OK en {elapsed:.1f}s ({size_mb:.0f} MB)")
    else:
        print(f"   ❌ Échec après {elapsed:.1f}s")
    
    return success


def process_temp(hour: str) -> bool:
    """Traite une image Temp."""
    src_path = os.path.join(CROPS_DIR, f"{hour}_Temp_crop.tif")
    dst_path = os.path.join(ALIGNED_DIR, f"{hour}_Temp_aligned.tif")
    
    print(f"\n🔄 Temp {hour}")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    t0 = time.time()
    
    # Cas spécial pour 13h
    if hour == REF_HOUR:
        transform = get_temp_transform_13h()
    else:
        transform = get_temp_transform(hour)
    
    print(f"   Application...")
    success = apply_transform(src_path, dst_path, transform)
    
    elapsed = time.time() - t0
    if success:
        size_mb = os.path.getsize(dst_path) / 1024 / 1024
        print(f"   ✅ OK en {elapsed:.1f}s ({size_mb:.0f} MB)")
    else:
        print(f"   ❌ Échec après {elapsed:.1f}s")
    
    return success


def copy_reference():
    """Copie les fichiers 13h de référence (pas de transformation)."""
    files = [
        ("13h_MS_crop.tif", "13h_MS_aligned.tif"),
        ("13h_NDVI.tif", "13h_NDVI_aligned.tif"),
    ]
    
    for src_name, dst_name in files:
        src_path = os.path.join(CROPS_DIR, src_name)
        dst_path = os.path.join(ALIGNED_DIR, dst_name)
        
        if os.path.exists(dst_path):
            print(f"📁 {dst_name} - déjà présent")
            continue
        
        print(f"📁 Copie {src_name} → {dst_name}")
        
        with rasterio.open(src_path) as src:
            data = src.read()
            profile = src.profile.copy()
            
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(data)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Applique les transformations à toutes les images"
    )
    parser.add_argument("--ms", action="store_true", help="Traiter les images MS")
    parser.add_argument("--ndvi", action="store_true", help="Traiter les images NDVI")
    parser.add_argument("--temp", action="store_true", help="Traiter les images Temp")
    parser.add_argument("--hour", type=str, help="Traiter une seule heure (ex: 09h)")
    parser.add_argument("--all", action="store_true", help="Tout traiter")
    
    args = parser.parse_args()
    
    # Si aucune option, afficher l'aide
    if not any([args.ms, args.ndvi, args.temp, args.all]):
        parser.print_help()
        return
    
    print("=" * 70)
    print("APPLICATION DES TRANSFORMATIONS")
    print("=" * 70)
    
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    
    # Déterminer les heures à traiter
    if args.hour:
        hours = [args.hour]
    else:
        hours = HOURS
    
    # Déterminer les types à traiter
    do_ms = args.ms or args.all
    do_ndvi = args.ndvi or args.all
    do_temp = args.temp or args.all
    
    # Copier les références
    if args.all:
        print("\n" + "=" * 70)
        print("FICHIERS DE RÉFÉRENCE (13h)")
        print("=" * 70)
        copy_reference()
    
    # Traiter chaque heure
    for hour in hours:
        print("\n" + "=" * 70)
        print(f"TRAITEMENT {hour}")
        print("=" * 70)
        
        if do_ms and hour != REF_HOUR:
            process_ms(hour)
        
        if do_ndvi and hour != REF_HOUR:
            process_ndvi(hour)
        
        if do_temp:
            process_temp(hour)
    
    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    
    print("\nFichiers créés:")
    for f in sorted(os.listdir(ALIGNED_DIR)):
        if f.endswith("_aligned.tif"):
            path = os.path.join(ALIGNED_DIR, f)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"   {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
