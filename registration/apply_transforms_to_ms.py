#!/usr/bin/env python3
"""
apply_transforms_to_ms.py - Applique les transformations aux images MS

Ce script applique les transformations calculées (NDVI → 13h-NDVI) aux images
multi-spectrales complètes (MS_crop) pour avoir toutes les bandes recalées.

Les transformations ont été calculées sur les NDVI, mais comme MS et NDVI
ont la même géométrie spatiale, on peut réutiliser les mêmes transformations.

Usage:
    python apply_transforms_to_ms.py
"""

import os
import sys
import numpy as np
import rasterio
from scipy import ndimage
import time

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from registration.nonlinear_registration import NonLinearTransform

# =============================================================================
# CONFIGURATION
# =============================================================================

CROPS_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/crops_2021"
ALIGNED_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/aligned_2021"

# Heures à traiter (13h est la référence, pas besoin de la transformer)
HOURS = ["09h", "11h", "15h", "17h"]

# Référence
REF_HOUR = "13h"


# =============================================================================
# FONCTIONS
# =============================================================================

def load_transform(hour: str) -> NonLinearTransform:
    """
    Charge la transformation pour une heure donnée.
    
    Args:
        hour: Heure (ex: "09h", "11h", etc.)
    
    Returns:
        NonLinearTransform chargé
    """
    # Format du nom de fichier (attention au format 9h vs 09h)
    hour_num = hour.replace("h", "").lstrip("0") or "0"
    prefix = f"transform_{hour_num}h_ndvi_to_13h_ndvi"
    
    base_path = os.path.join(ALIGNED_DIR, prefix)
    
    print(f"   Chargement transformation: {prefix}")
    
    # Utiliser la méthode load de NonLinearTransform
    transform = NonLinearTransform.load(base_path)
    return transform


def apply_transform_to_ms(hour: str, transform: NonLinearTransform) -> str:
    """
    Applique une transformation à l'image MS d'une heure donnée.
    
    Args:
        hour: Heure (ex: "09h")
        transform: Transformation à appliquer
    
    Returns:
        Chemin du fichier créé
    """
    src_path = os.path.join(CROPS_DIR, f"{hour}_MS_crop.tif")
    dst_path = os.path.join(ALIGNED_DIR, f"{hour}_MS_aligned.tif")
    
    print(f"\n🔄 Transformation {hour} MS")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    t0 = time.time()
    
    # Charger le profil de référence
    ref_path = os.path.join(ALIGNED_DIR, "13h_NDVI.tif")
    with rasterio.open(ref_path) as ref:
        ref_profile = ref.profile.copy()
    
    with rasterio.open(src_path) as src:
        n_bands = src.count
        height, width = src.height, src.width
        print(f"   Taille source: {width} x {height} pixels, {n_bands} bandes")
        print(f"   Taille sortie: {ref_profile['width']} x {ref_profile['height']} pixels")
    
    # Utiliser la méthode apply_to_file de NonLinearTransform
    print(f"   Application de la transformation...")
    success = transform.apply_to_file(src_path, dst_path, ref_profile=ref_profile)
    
    elapsed = time.time() - t0
    if success:
        print(f"✅ Fichier créé en {elapsed:.1f}s: {dst_path}")
    else:
        print(f"❌ Échec après {elapsed:.1f}s")
    
    return dst_path if success else None


def copy_reference_ms():
    """Copie le MS 13h comme référence (pas de transformation nécessaire)."""
    src_path = os.path.join(CROPS_DIR, f"{REF_HOUR}_MS_crop.tif")
    dst_path = os.path.join(ALIGNED_DIR, f"{REF_HOUR}_MS_aligned.tif")
    
    if os.path.exists(dst_path):
        print(f"\n📁 {REF_HOUR} MS (référence) - déjà présent")
        return dst_path
    
    print(f"\n📁 Copie {REF_HOUR} MS (référence)")
    print(f"   Source: {src_path}")
    print(f"   Destination: {dst_path}")
    
    # Copie simple avec rasterio pour garder les métadonnées
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        data = src.read()
        
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(data)
    
    print(f"✅ Fichier copié: {dst_path}")
    return dst_path


def main():
    print("=" * 70)
    print("APPLICATION DES TRANSFORMATIONS AUX IMAGES MS")
    print("=" * 70)
    print(f"\nRéférence: {REF_HOUR}")
    print(f"Heures à transformer: {HOURS}")
    
    # Vérifier que le dossier de sortie existe
    os.makedirs(ALIGNED_DIR, exist_ok=True)
    
    # Copier la référence
    copy_reference_ms()
    
    # Transformer chaque heure
    for hour in HOURS:
        try:
            print(f"\n{'='*70}")
            print(f"TRAITEMENT {hour}")
            print("=" * 70)
            
            # Charger la transformation
            transform = load_transform(hour)
            
            # Appliquer au MS
            apply_transform_to_ms(hour, transform)
            
        except Exception as e:
            print(f"❌ Erreur pour {hour}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TERMINÉ")
    print("=" * 70)
    
    # Résumé
    print("\nFichiers créés:")
    for f in sorted(os.listdir(ALIGNED_DIR)):
        if f.endswith("_MS_aligned.tif"):
            path = os.path.join(ALIGNED_DIR, f)
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"   {f} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
