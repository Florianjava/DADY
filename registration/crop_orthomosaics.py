#!/usr/bin/env python3
"""
crop_orthomosaics.py - Applique les crops définis dans crop_coordinates.csv

Lit le fichier CSV avec les coordonnées de crop et exécute gdal_translate
pour chaque orthomosaïque.

Usage:
    python -m registration.crop_orthomosaics
"""

import os
import pandas as pd
import subprocess
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoire source des orthomosaïques originales
SOURCE_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/anatomics"

# Répertoire de travail pour Fijiyama
WORK_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"

# Fichier CSV avec les coordonnées
CSV_PATH = os.path.join(WORK_DIR, "crop_coordinates.csv")

# Dossier de sortie pour les crops
OUTPUT_DIR = os.path.join(WORK_DIR, "crops_2021")

# Mapping type -> suffixe fichier source
TYPE_TO_SUFFIX = {
    "MS": "_MS.tif",      # Pour NDVI, on utilise l'image MS
    "Temp": "_IR.tif"     # Pour température, on utilise l'image IR
}

# Mapping type -> suffixe fichier sortie
TYPE_TO_OUTPUT = {
    "MS": "_NDVI_crop.tif",
    "Temp": "_Temp_crop.tif"
}


# =============================================================================
# FONCTIONS
# =============================================================================

def load_crop_coordinates(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Charge les coordonnées de crop depuis le CSV."""
    df = pd.read_csv(csv_path)
    print(f"📄 {len(df)} crops à effectuer")
    return df


def get_source_path(hour: str, crop_type: str, source_dir: str = SOURCE_DIR) -> str:
    """
    Retourne le chemin du fichier source.
    
    Gère les variantes de nommage (09h vs 9h).
    """
    suffix = TYPE_TO_SUFFIX[crop_type]
    
    # Essayer les deux variantes
    for h in [hour, hour.lstrip("0")]:
        path = os.path.join(source_dir, f"2021_{h}{suffix}")
        if os.path.exists(path):
            return path
    
    return None


def crop_with_gdal(
    src_path: str,
    dst_path: str,
    xoff: int,
    yoff: int,
    xsize: int,
    ysize: int,
    verbose: bool = True
) -> bool:
    """
    Exécute gdal_translate pour cropper une image.
    
    Args:
        src_path: Fichier source
        dst_path: Fichier destination
        xoff, yoff: Décalage en pixels
        xsize, ysize: Taille du crop en pixels
        verbose: Afficher les commandes
        
    Returns:
        True si succès
    """
    cmd = [
        "gdal_translate",
        "-srcwin", str(xoff), str(yoff), str(xsize), str(ysize),
        src_path,
        dst_path
    ]
    
    if verbose:
        print(f"   $ {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ Erreur: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False


def crop_all_orthomosaics(
    csv_path: str = CSV_PATH,
    source_dir: str = SOURCE_DIR,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True
) -> dict:
    """
    Effectue tous les crops définis dans le CSV.
    
    Returns:
        Rapport d'exécution
    """
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Charger les coordonnées
    df = load_crop_coordinates(csv_path)
    
    report = {"success": [], "errors": []}
    
    print(f"\n{'='*60}")
    print("CROP DES ORTHOMOSAÏQUES")
    print(f"{'='*60}")
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print()
    
    for _, row in df.iterrows():
        hour = row["hour"]
        crop_type = row["type"]
        xoff = int(row["xoff"])
        yoff = int(row["yoff"])
        xsize = int(row["xsize"])
        ysize = int(row["ysize"])
        
        if verbose:
            print(f"📷 {hour} {crop_type}: ({xoff}, {yoff}) -> {xsize}x{ysize}")
        
        # Trouver le fichier source
        src_path = get_source_path(hour, crop_type, source_dir)
        if src_path is None:
            print(f"   ⚠️ Fichier source non trouvé")
            report["errors"].append(f"{hour}_{crop_type}")
            continue
        
        # Fichier de sortie
        output_suffix = TYPE_TO_OUTPUT[crop_type]
        dst_path = os.path.join(output_dir, f"{hour}{output_suffix}")
        
        # Exécuter le crop
        success = crop_with_gdal(src_path, dst_path, xoff, yoff, xsize, ysize, verbose=verbose)
        
        if success:
            # Vérifier la taille du fichier créé
            if os.path.exists(dst_path):
                size_mb = os.path.getsize(dst_path) / (1024 * 1024)
                print(f"   ✅ {size_mb:.1f} MB")
            report["success"].append(f"{hour}_{crop_type}")
        else:
            report["errors"].append(f"{hour}_{crop_type}")
    
    print(f"\n{'='*60}")
    print(f"✅ {len(report['success'])} crops réussis")
    if report["errors"]:
        print(f"❌ {len(report['errors'])} erreurs: {report['errors']}")
    print(f"{'='*60}")
    
    return report


def compute_ndvi_from_ms_crop(
    ms_crop_path: str,
    ndvi_output_path: str,
    red_band: int = 3,
    nir_band: int = 5
) -> bool:
    """
    Calcule le NDVI à partir d'un crop MS.
    
    Args:
        ms_crop_path: Chemin du crop MS (6 bandes)
        ndvi_output_path: Chemin de sortie pour le NDVI
        red_band: Numéro de la bande Red (1-indexed)
        nir_band: Numéro de la bande NIR (1-indexed)
        
    Returns:
        True si succès
    """
    import rasterio
    import numpy as np
    
    try:
        with rasterio.open(ms_crop_path) as src:
            red = src.read(red_band).astype(np.float32)
            nir = src.read(nir_band).astype(np.float32)
            profile = src.profile.copy()
        
        # Calcul NDVI
        with np.errstate(divide='ignore', invalid='ignore'):
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
        
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
        profile.update(dtype=np.float32, count=1)
        
        with rasterio.open(ndvi_output_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur NDVI: {e}")
        return False


def process_all_crops(output_dir: str = OUTPUT_DIR, verbose: bool = True) -> dict:
    """
    Après le crop MS, calcule les NDVI et mean_MS pour Fijiyama.
    """
    import rasterio
    import numpy as np
    
    report = {"ndvi": [], "mean_ms": []}
    
    print(f"\n{'='*60}")
    print("CALCUL DES PRODUITS DÉRIVÉS")
    print(f"{'='*60}")
    
    # Trouver tous les crops MS
    for f in os.listdir(output_dir):
        if f.endswith("_NDVI_crop.tif"):
            # C'est en fait un crop MS, on doit calculer le vrai NDVI
            hour = f.split("_")[0]
            ms_path = os.path.join(output_dir, f)
            
            if verbose:
                print(f"\n📷 {hour}:")
            
            # Vérifier si c'est bien une image multi-bandes
            with rasterio.open(ms_path) as src:
                n_bands = src.count
            
            if n_bands >= 5:
                # C'est un MS, calculer NDVI et mean
                
                # 1. NDVI
                ndvi_path = os.path.join(output_dir, f"{hour}_NDVI.tif")
                if compute_ndvi_from_ms_crop(ms_path, ndvi_path):
                    if verbose:
                        print(f"   NDVI ✅")
                    report["ndvi"].append(hour)
                
                # 2. Mean MS
                mean_path = os.path.join(output_dir, f"{hour}_mean_MS.tif")
                try:
                    with rasterio.open(ms_path) as src:
                        data = src.read().astype(np.float32)
                        profile = src.profile.copy()
                    
                    mean_data = np.mean(data, axis=0)
                    profile.update(dtype=np.float32, count=1)
                    
                    with rasterio.open(mean_path, 'w', **profile) as dst:
                        dst.write(mean_data, 1)
                    
                    if verbose:
                        print(f"   mean_MS ✅")
                    report["mean_ms"].append(hour)
                    
                except Exception as e:
                    print(f"   mean_MS ❌: {e}")
                
                # Renommer le crop MS original
                ms_renamed = os.path.join(output_dir, f"{hour}_MS_crop.tif")
                if not os.path.exists(ms_renamed):
                    os.rename(ms_path, ms_renamed)
                    if verbose:
                        print(f"   Renommé: {f} -> {hour}_MS_crop.tif")
    
    return report


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    """Point d'entrée principal."""
    
    # Étape 1: Crop des orthomosaïques
    report_crop = crop_all_orthomosaics()
    
    # Étape 2: Calcul NDVI et mean_MS
    report_products = process_all_crops()
    
    print(f"\n{'='*60}")
    print("FICHIERS PRÊTS POUR FIJIYAMA")
    print(f"{'='*60}")
    print(f"Dossier: {OUTPUT_DIR}")
    print()
    print("Pour le recalage, utiliser les fichiers *_mean_MS.tif")
    print("L'image de référence est 13h_mean_MS.tif")
    print()
    print("Après recalage, forcer le géoréférencement de 13h sur les autres heures.")


if __name__ == "__main__":
    main()
