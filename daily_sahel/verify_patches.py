#!/usr/bin/env python3
"""
verify_patches.py - Vérification et debug des patches exportés

Ce script permet de vérifier l'intégrité et la cohérence des patches exportés :
- Vérification de la structure des fichiers
- Lecture et affichage des statistiques d'un patch
- Comparaison entre patches à différentes heures

Usage:
    python -m daily_sahel.verify_patches
    
    ou en important:
    from daily_sahel.verify_patches import check_patch, compare_patch_hours
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

try:
    import rasterio
except ImportError:
    rasterio = None


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_PATCHES_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed"
HOURS = ["09h", "11h", "13h", "15h", "17h"]
BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6", "NDVI", "Temp"]
MASK_NAMES = ["mask_poly", "mask_plant", "mask_soil"]


# =============================================================================
# FONCTIONS DE VÉRIFICATION
# =============================================================================

def check_directory_structure(patches_dir: str = DEFAULT_PATCHES_DIR) -> Dict:
    """
    Vérifie la structure du répertoire de patches.
    
    Returns:
        Dictionnaire avec le compte des fichiers par heure
    """
    results = {"ok": True, "hours": {}, "csv_exists": False}
    
    # Vérifier le CSV général
    csv_path = os.path.join(patches_dir, "daily_sahel_2021_complete.csv")
    results["csv_exists"] = os.path.exists(csv_path)
    
    # Vérifier chaque heure
    for hour in HOURS:
        hour_dir = os.path.join(patches_dir, f"2021_{hour}")
        if os.path.exists(hour_dir):
            files = os.listdir(hour_dir)
            results["hours"][hour] = {
                "exists": True,
                "file_count": len(files),
                "expected": 640 * 12  # 640 plots × 12 fichiers
            }
            if len(files) != 640 * 12:
                results["ok"] = False
        else:
            results["hours"][hour] = {"exists": False}
            results["ok"] = False
    
    return results


def list_patches_for_plot(plot: str, patches_dir: str = DEFAULT_PATCHES_DIR) -> Dict:
    """
    Liste tous les fichiers disponibles pour un plot donné.
    
    Args:
        plot: Identifiant du plot (ex: "Pa1", "Pa42")
        patches_dir: Répertoire des patches
        
    Returns:
        Dictionnaire {heure: liste_fichiers}
    """
    results = {}
    
    for hour in HOURS:
        hour_dir = os.path.join(patches_dir, f"2021_{hour}")
        if os.path.exists(hour_dir):
            pattern = f"{plot}_{hour}_"
            files = [f for f in os.listdir(hour_dir) if f.startswith(pattern)]
            results[hour] = sorted(files)
    
    return results


def check_patch(plot: str, hour: str, patches_dir: str = DEFAULT_PATCHES_DIR) -> Dict:
    """
    Vérifie et lit les données d'un patch spécifique.
    
    Args:
        plot: Identifiant du plot (ex: "Pa1")
        hour: Heure (ex: "11h")
        patches_dir: Répertoire des patches
        
    Returns:
        Dictionnaire avec les statistiques du patch
    """
    if rasterio is None:
        return {"error": "rasterio non installé"}
    
    hour_dir = os.path.join(patches_dir, f"2021_{hour}")
    prefix = f"{plot}_{hour}"
    
    results = {
        "plot": plot,
        "hour": hour,
        "bands": {},
        "masks": {},
        "info": None
    }
    
    # Lire les bandes
    for band in BAND_NAMES:
        band_path = os.path.join(hour_dir, f"{prefix}_{band}.tif")
        if os.path.exists(band_path):
            with rasterio.open(band_path) as src:
                data = src.read(1)
                results["bands"][band] = {
                    "shape": data.shape,
                    "dtype": str(data.dtype),
                    "min": float(np.nanmin(data)),
                    "max": float(np.nanmax(data)),
                    "mean": float(np.nanmean(data)),
                    "std": float(np.nanstd(data))
                }
    
    # Lire les masques
    for mask in MASK_NAMES:
        mask_path = os.path.join(hour_dir, f"{prefix}_{mask}.tif")
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as src:
                data = src.read(1)
                results["masks"][mask] = {
                    "shape": data.shape,
                    "pixels_total": int(data.size),
                    "pixels_masked": int(np.sum(data > 0)),
                    "coverage_pct": float(np.sum(data > 0) / data.size * 100)
                }
    
    # Lire le CSV info
    info_path = os.path.join(hour_dir, f"{prefix}_info.csv")
    if os.path.exists(info_path):
        results["info"] = pd.read_csv(info_path).iloc[0].to_dict()
    
    return results


def compare_patch_hours(plot: str, patches_dir: str = DEFAULT_PATCHES_DIR) -> pd.DataFrame:
    """
    Compare les statistiques d'un patch à travers les différentes heures.
    
    Args:
        plot: Identifiant du plot
        patches_dir: Répertoire des patches
        
    Returns:
        DataFrame avec l'évolution par heure
    """
    rows = []
    
    for hour in HOURS:
        patch_info = check_patch(plot, hour, patches_dir)
        
        if patch_info.get("info"):
            row = {
                "hour": hour,
                "NDVI_poly": patch_info["info"].get("mean_NDVI_poly"),
                "NDVI_plant": patch_info["info"].get("mean_NDVI_plant"),
                "Temp_poly": patch_info["info"].get("mean_Temp_poly"),
                "surface_plant": patch_info["info"].get("surface_plant"),
                "surface_poly": patch_info["info"].get("surface_poly"),
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def verify_consistency(patches_dir: str = DEFAULT_PATCHES_DIR, 
                       sample_size: int = 10) -> Dict:
    """
    Vérifie la cohérence entre les CSV info et les masques.
    
    Args:
        patches_dir: Répertoire des patches
        sample_size: Nombre de patches à vérifier par heure
        
    Returns:
        Rapport de vérification
    """
    import random
    
    results = {"verified": 0, "errors": [], "ok": True}
    
    # Charger le CSV général
    csv_path = os.path.join(patches_dir, "daily_sahel_2021_complete.csv")
    df = pd.read_csv(csv_path)
    
    for hour in HOURS:
        hour_df = df[df['hour'] == hour]
        sample_plots = random.sample(list(hour_df['plot_excel'].unique()), 
                                     min(sample_size, len(hour_df)))
        
        for plot in sample_plots:
            patch = check_patch(plot, hour, patches_dir)
            
            # Vérifier surface_poly
            if patch.get("masks", {}).get("mask_poly"):
                mask_pixels = patch["masks"]["mask_poly"]["pixels_masked"]
                csv_pixels = patch["info"]["surface_poly"]
                
                if abs(mask_pixels - csv_pixels) > 1:
                    results["errors"].append({
                        "plot": plot,
                        "hour": hour,
                        "issue": f"surface_poly mismatch: mask={mask_pixels}, csv={csv_pixels}"
                    })
                    results["ok"] = False
            
            results["verified"] += 1
    
    return results


def print_patch_report(plot: str, hour: str, patches_dir: str = DEFAULT_PATCHES_DIR) -> None:
    """Affiche un rapport détaillé pour un patch."""
    patch = check_patch(plot, hour, patches_dir)
    
    print("=" * 70)
    print(f"RAPPORT PATCH: {plot} @ {hour}")
    print("=" * 70)
    
    if patch.get("info"):
        info = patch["info"]
        print(f"\n📋 INFORMATIONS GÉNÉRALES:")
        print(f"   Génotype: {info.get('geno')}")
        print(f"   Environnement: {info.get('env')}")
        print(f"   Répétition: {info.get('repetition')}")
        print(f"   Bloc: {info.get('bloc')}")
    
    print(f"\n📊 BANDES SPECTRALES:")
    for band, stats in patch.get("bands", {}).items():
        print(f"   {band:10s}: shape={stats['shape']}, "
              f"min={stats['min']:.1f}, max={stats['max']:.1f}, "
              f"mean={stats['mean']:.1f}")
    
    print(f"\n🎭 MASQUES:")
    for mask, stats in patch.get("masks", {}).items():
        print(f"   {mask:12s}: {stats['pixels_masked']:5d} pixels "
              f"({stats['coverage_pct']:.1f}%)")
    
    if patch.get("info"):
        print(f"\n📈 MESURES:")
        print(f"   NDVI (poly):  {info.get('mean_NDVI_poly', 'N/A'):.4f}")
        print(f"   NDVI (plant): {info.get('mean_NDVI_plant', 'N/A'):.4f}")
        print(f"   Temp (poly):  {info.get('mean_Temp_poly', 'N/A'):.1f}")


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    """Vérification complète des patches."""
    print("=" * 70)
    print("VÉRIFICATION DES PATCHES EXPORTÉS")
    print("=" * 70)
    
    # 1. Structure des répertoires
    print("\n1. STRUCTURE DES RÉPERTOIRES")
    print("-" * 40)
    structure = check_directory_structure()
    print(f"   CSV général: {'✓' if structure['csv_exists'] else '✗'}")
    for hour, info in structure["hours"].items():
        if info.get("exists"):
            status = "✓" if info["file_count"] == info["expected"] else "⚠️"
            print(f"   {hour}: {status} {info['file_count']}/{info['expected']} fichiers")
        else:
            print(f"   {hour}: ✗ répertoire manquant")
    
    # 2. Exemple de patch
    print("\n2. EXEMPLE DE PATCH (Pa1 @ 11h)")
    print("-" * 40)
    print_patch_report("Pa1", "11h")
    
    # 3. Évolution journalière
    print("\n3. ÉVOLUTION JOURNALIÈRE (Pa1)")
    print("-" * 40)
    evolution = compare_patch_hours("Pa1")
    print(evolution.to_string(index=False))
    
    # 4. Vérification de cohérence
    print("\n4. VÉRIFICATION DE COHÉRENCE (échantillon)")
    print("-" * 40)
    consistency = verify_consistency(sample_size=5)
    print(f"   Patches vérifiés: {consistency['verified']}")
    print(f"   Statut: {'✓ OK' if consistency['ok'] else '✗ ERREURS'}")
    if consistency["errors"]:
        for err in consistency["errors"][:5]:
            print(f"   ⚠️ {err['plot']}@{err['hour']}: {err['issue']}")
    
    print("\n" + "=" * 70)
    print("VÉRIFICATION TERMINÉE")
    print("=" * 70)


if __name__ == "__main__":
    main()
