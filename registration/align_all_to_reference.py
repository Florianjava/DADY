#!/usr/bin/env python3
"""
align_all_to_reference.py - Recale toutes les images sur 13h-NDVI

Ce script recale toutes les images (NDVI, MS et Temp) de toutes les heures
sur la référence commune : 13h-NDVI.

Pour les NDVI et MS (même géométrie) :
    Xh-NDVI/MS → 13h-NDVI (transformation directe)

Pour les Temp :
    Xh-Temp → 13h-Temp → 13h-NDVI (composition de deux transformations)

Usage:
    python align_all_to_reference.py [--hours 9,11,15,17] [--dry-run]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import rasterio

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.nonlinear_registration import (
    NonLinearTransform,
    compose_transforms
)


# =============================================================================
# CONFIGURATION
# =============================================================================

WORK_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"
CROPS_DIR = os.path.join(WORK_DIR, "crops_2021")
ROIS_DIR = os.path.join(WORK_DIR, "Rois_0_align_crop_affine")
OUTPUT_DIR = os.path.join(WORK_DIR, "aligned_2021")

# Heures disponibles (hors 13h qui est la référence pour NDVI)
# Note: Les ROI utilisent "9", "11", etc. mais les fichiers images utilisent "09h", "11h", etc.
HOURS = ["9", "11", "15", "17"]


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def format_hour_for_file(hour: str) -> str:
    """
    Formate l'heure pour les noms de fichiers images (avec zéro devant si < 10).
    Ex: "9" -> "09", "11" -> "11"
    """
    return hour.zfill(2)


def load_points(roi_prefix: str) -> tuple:
    """
    Charge les points de correspondance depuis les fichiers CSV.
    
    Args:
        roi_prefix: Préfixe du fichier ROI (ex: "RoiSet_9-ndvi_to_13-ndvi")
    
    Returns:
        (pts_ref, pts_src) : Points de référence et source
    """
    ref_csv = os.path.join(ROIS_DIR, f"{roi_prefix}_set1.csv")
    src_csv = os.path.join(ROIS_DIR, f"{roi_prefix}_set2.csv")
    
    if not os.path.exists(ref_csv):
        raise FileNotFoundError(f"Fichier non trouvé: {ref_csv}")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Fichier non trouvé: {src_csv}")
    
    pts_ref = pd.read_csv(ref_csv)[['x', 'y']].values
    pts_src = pd.read_csv(src_csv)[['x', 'y']].values
    
    return pts_ref, pts_src


def get_reference_info():
    """Charge les infos de l'image de référence (13h-NDVI)."""
    ref_path = os.path.join(CROPS_DIR, "13h_NDVI.tif")
    with rasterio.open(ref_path) as src:
        return {
            'shape': (src.height, src.width),
            'profile': src.profile.copy(),
            'path': ref_path
        }


def compute_transform(hour: str, modality: str, ref_info: dict) -> NonLinearTransform:
    """
    Calcule la transformation pour une image donnée vers 13h-NDVI.
    
    Args:
        hour: Heure ("9", "11", "13", "15", "17")
        modality: "ndvi" ou "temp"
        ref_info: Infos de l'image de référence
    
    Returns:
        NonLinearTransform configurée
    """
    output_shape = ref_info['shape']
    
    if modality == "ndvi":
        if hour == "13":
            # 13h-NDVI est la référence, pas de transformation
            t = NonLinearTransform()
            t.affine_matrix = np.eye(3)
            t.output_shape = output_shape
            t.mode = 'affine'
            return t
        
        # Transformation directe : Xh-NDVI → 13h-NDVI
        # Format: RoiSet_X-ndvi_to_13-ndvi
        roi_prefix = f"RoiSet_{hour}-ndvi_to_13-ndvi"
        pts_ref, pts_src = load_points(roi_prefix)
        
        print(f"\n{'='*60}")
        print(f"📐 {hour}h-NDVI → 13h-NDVI (direct)")
        print(f"   {len(pts_ref)} points de correspondance")
        print(f"{'='*60}")
        
        t = NonLinearTransform()
        t.compute_from_points(pts_ref, pts_src, output_shape, mode='affine+nonlinear')
        return t
    
    elif modality == "temp":
        # Pour Temp : composition de deux transformations
        # 1. Xh-Temp → 13h-Temp
        # 2. 13h-Temp → 13h-NDVI
        
        print(f"\n{'='*60}")
        print(f"📐 {hour}h-Temp → 13h-NDVI (composition)")
        print(f"{'='*60}")
        
        # Transformation 1 : Xh-Temp → 13h-Temp
        if hour == "13":
            # Pas de première transformation
            t1 = NonLinearTransform()
            t1.affine_matrix = np.eye(3)
            t1.output_shape = output_shape
            t1.mode = 'affine'
            print(f"   Étape 1: 13h-Temp (identité)")
        else:
            # Format: RoiSet_X-temp_to_13-temp
            roi_prefix_1 = f"RoiSet_{hour}-temp_to_13-temp"
            pts_ref_1, pts_src_1 = load_points(roi_prefix_1)
            print(f"   Étape 1: {hour}h-Temp → 13h-Temp ({len(pts_ref_1)} points)")
            
            # Pour cette transformation, on a besoin de la taille de 13h-Temp
            temp_13_path = os.path.join(CROPS_DIR, "13h_Temp_crop.tif")
            with rasterio.open(temp_13_path) as src:
                temp_13_shape = (src.height, src.width)
            
            t1 = NonLinearTransform()
            t1.compute_from_points(pts_ref_1, pts_src_1, temp_13_shape, mode='affine+nonlinear')
        
        # Transformation 2 : 13h-Temp → 13h-NDVI
        # Format: RoiSet_13-temp_to_13-ndvi
        roi_prefix_2 = "RoiSet_13-temp_to_13-ndvi"
        pts_ref_2, pts_src_2 = load_points(roi_prefix_2)
        print(f"   Étape 2: 13h-Temp → 13h-NDVI ({len(pts_ref_2)} points)")
        
        t2 = NonLinearTransform()
        t2.compute_from_points(pts_ref_2, pts_src_2, output_shape, mode='affine+nonlinear')
        
        # Composition : result = t2 ∘ t1
        if hour == "13":
            return t2
        else:
            return compose_transforms(t1, t2, output_shape)
    
    else:
        raise ValueError(f"modality doit être 'ndvi' ou 'temp', pas '{modality}'")


def apply_and_save(
    transform: NonLinearTransform,
    src_path: str,
    dst_path: str,
    ref_profile: dict
) -> bool:
    """Applique une transformation et sauvegarde le résultat."""
    return transform.apply_to_file(src_path, dst_path, ref_profile)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================

def align_all(hours: list = None, dry_run: bool = False):
    """
    Recale toutes les images sur 13h-NDVI.
    
    Args:
        hours: Liste des heures à traiter (si None, toutes)
        dry_run: Si True, ne fait que calculer sans sauvegarder
    """
    if hours is None:
        hours = HOURS
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Charger les infos de référence
    ref_info = get_reference_info()
    print(f"\n🎯 Référence: 13h-NDVI")
    print(f"   Taille: {ref_info['shape'][1]} x {ref_info['shape'][0]}")
    
    results = []
    
    # =========================================================================
    # TRAITER LES NDVI
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# IMAGES NDVI")
    print(f"{'#'*60}")
    
    for hour in hours:
        try:
            transform = compute_transform(hour, "ndvi", ref_info)
            
            hour_fmt = format_hour_for_file(hour)
            src_path = os.path.join(CROPS_DIR, f"{hour_fmt}h_NDVI.tif")
            dst_path = os.path.join(OUTPUT_DIR, f"{hour_fmt}h_NDVI_aligned.tif")
            
            if not dry_run:
                success = apply_and_save(transform, src_path, dst_path, ref_info['profile'])
                if success:
                    print(f"✅ {hour}h-NDVI → {dst_path}")
                    results.append((hour, "ndvi", "OK"))
                else:
                    print(f"❌ {hour}h-NDVI : échec")
                    results.append((hour, "ndvi", "FAIL"))
            else:
                print(f"🔍 [dry-run] {hour}h-NDVI serait sauvé dans {dst_path}")
                results.append((hour, "ndvi", "DRY"))
            
            # Sauvegarder la transformation
            if not dry_run:
                transform_path = os.path.join(OUTPUT_DIR, f"transform_{hour}h_ndvi_to_13h_ndvi")
                transform.save(transform_path)
                
        except Exception as e:
            print(f"❌ {hour}h-NDVI : {e}")
            results.append((hour, "ndvi", f"ERROR: {e}"))
    
    # =========================================================================
    # TRAITER LES TEMP
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# IMAGES TEMP")
    print(f"{'#'*60}")
    
    # Ajouter 13h pour les Temp (doit aussi être recalé sur 13h-NDVI)
    temp_hours = ["13"] + hours
    
    for hour in temp_hours:
        try:
            transform = compute_transform(hour, "temp", ref_info)
            
            hour_fmt = format_hour_for_file(hour)
            src_path = os.path.join(CROPS_DIR, f"{hour_fmt}h_Temp_crop.tif")
            dst_path = os.path.join(OUTPUT_DIR, f"{hour_fmt}h_Temp_aligned.tif")
            
            if not dry_run:
                success = apply_and_save(transform, src_path, dst_path, ref_info['profile'])
                if success:
                    print(f"✅ {hour}h-Temp → {dst_path}")
                    results.append((hour, "temp", "OK"))
                else:
                    print(f"❌ {hour}h-Temp : échec")
                    results.append((hour, "temp", "FAIL"))
            else:
                print(f"🔍 [dry-run] {hour}h-Temp serait sauvé dans {dst_path}")
                results.append((hour, "temp", "DRY"))
            
            # Sauvegarder la transformation
            if not dry_run:
                transform_path = os.path.join(OUTPUT_DIR, f"transform_{hour}h_temp_to_13h_ndvi")
                transform.save(transform_path)
                
        except Exception as e:
            print(f"❌ {hour}h-Temp : {e}")
            results.append((hour, "temp", f"ERROR: {e}"))
    
    # =========================================================================
    # TRAITER LES MS (même transformation que NDVI)
    # =========================================================================
    print(f"\n{'#'*60}")
    print(f"# IMAGES MS (multi-spectral)")
    print(f"{'#'*60}")
    
    # Ajouter 13h pour les MS
    ms_hours = ["13"] + hours
    
    for hour in ms_hours:
        try:
            hour_fmt = format_hour_for_file(hour)
            src_path = os.path.join(CROPS_DIR, f"{hour_fmt}h_MS_crop.tif")
            dst_path = os.path.join(OUTPUT_DIR, f"{hour_fmt}h_MS_aligned.tif")
            
            if hour == "13":
                # 13h-MS : copie simple (référence)
                print(f"\n📐 13h-MS (référence, copie directe)")
                if not dry_run:
                    with rasterio.open(src_path) as src:
                        data = src.read()
                        profile = src.profile.copy()
                        # Garder la même taille (pas de resampling)
                        with rasterio.open(dst_path, 'w', **profile) as dst:
                            dst.write(data)
                    print(f"✅ 13h-MS → {dst_path}")
                    results.append(("13", "ms", "OK"))
                else:
                    print(f"🔍 [dry-run] 13h-MS serait copié dans {dst_path}")
                    results.append(("13", "ms", "DRY"))
            else:
                # Charger la transformation NDVI déjà calculée
                transform_path = os.path.join(OUTPUT_DIR, f"transform_{hour}h_ndvi_to_13h_ndvi")
                print(f"\n📐 {hour}h-MS → 13h-NDVI (réutilise transfo NDVI)")
                
                transform = NonLinearTransform.load(transform_path)
                
                if not dry_run:
                    success = apply_and_save(transform, src_path, dst_path, ref_info['profile'])
                    if success:
                        print(f"✅ {hour}h-MS → {dst_path}")
                        results.append((hour, "ms", "OK"))
                    else:
                        print(f"❌ {hour}h-MS : échec")
                        results.append((hour, "ms", "FAIL"))
                else:
                    print(f"🔍 [dry-run] {hour}h-MS serait sauvé dans {dst_path}")
                    results.append((hour, "ms", "DRY"))
                
        except Exception as e:
            print(f"❌ {hour}h-MS : {e}")
            results.append((hour, "ms", f"ERROR: {e}"))
    
    # =========================================================================
    # RÉSUMÉ
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ")
    print(f"{'='*60}")
    print(f"{'Heure':<10} {'Modalité':<10} {'Statut':<20}")
    print("-" * 40)
    for hour, modality, status in results:
        print(f"{hour:<10} {modality:<10} {status:<20}")
    
    # Compter les succès
    ok_count = sum(1 for _, _, s in results if s == "OK")
    total = len(results)
    print(f"\n✅ {ok_count}/{total} images traitées avec succès")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recale toutes les images sur 13h-NDVI")
    parser.add_argument(
        "--hours",
        type=str,
        default=None,
        help="Heures à traiter, séparées par des virgules (ex: 9,11,15,17)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Ne fait que calculer sans sauvegarder"
    )
    
    args = parser.parse_args()
    
    hours = None
    if args.hours:
        hours = [h.strip() for h in args.hours.split(",")]
    
    align_all(hours=hours, dry_run=args.dry_run)

