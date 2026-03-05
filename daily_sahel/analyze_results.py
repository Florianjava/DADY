"""
analyze_results.py - Analyse des patches extraits avec segmentation

Ce script analyse les patches extraits pour chaque plot :
1. Charge les canaux MS normalisés (déjà normalisés par les moquettes)
2. Calcule une segmentation des plantes (NDVI > 0 basé sur les MS normalisés)
3. Extrait les statistiques globales, plantes et non-plantes
4. Génère un CSV par plot avec une ligne par heure

Les MS normalisés sont calculés lors de l'extraction (extract_aligned_2021.py)
avec les coefficients de normalisation basés sur les moquettes (référence = 9h).

Usage:
    python -m daily_sahel.analyze_results --plot ETM_Rep1_001
    python -m daily_sahel.analyze_results --all
    python -m daily_sahel.analyze_results --all --max-plots 10
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import warnings

# Supprimer les warnings rasterio
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoires
PATCHES_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/patches_aligned_2021"
OUTPUT_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/analysis_results"

# Heures disponibles
HOURS = ["09h", "11h", "13h", "15h", "17h"]

# Noms des bandes MS
MS_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6"]

# Seuil NDVI pour la segmentation des plantes
NDVI_THRESHOLD = 0.0


# =============================================================================
# ANALYSE D'UN PLOT
# =============================================================================

def analyze_plot(plot_name: str, save_segmentation: bool = True) -> List[Dict]:
    """
    Analyse un plot pour toutes les heures.
    
    Utilise les MS normalisés (déjà calculés lors de l'extraction).
    
    Args:
        plot_name: Nom du plot (ex: "ETM_Rep1_001")
        save_segmentation: Sauvegarder les images de segmentation
    
    Returns:
        Liste de dicts (un par heure) avec les statistiques
    """
    plot_dir = os.path.join(PATCHES_DIR, plot_name)
    
    if not os.path.exists(plot_dir):
        print(f"⚠️ Dossier non trouvé: {plot_dir}")
        return []
    
    # Charger le masque du polygone
    mask_path = os.path.join(plot_dir, f"{plot_name}_mask.tif")
    if not os.path.exists(mask_path):
        print(f"⚠️ Masque non trouvé: {mask_path}")
        return []
    
    with rasterio.open(mask_path) as src:
        mask = src.read(1)  # 1 = dans le polygone, 0 = dehors
    
    # Créer le dossier Segmentation si nécessaire
    if save_segmentation:
        seg_dir = os.path.join(plot_dir, "Segmentation")
        os.makedirs(seg_dir, exist_ok=True)
    
    results = []
    
    for hour in HOURS:
        result = {"plot_name": plot_name, "hour": hour}
        
        # Chemins des fichiers
        ms_norm_path = os.path.join(plot_dir, "MS_normalized", f"{plot_name}_{hour}_MS_normalized.tif")
        ms_path = os.path.join(plot_dir, "MS", f"{plot_name}_{hour}_MS.tif")
        temp_path = os.path.join(plot_dir, "Temp", f"{plot_name}_{hour}_Temp.tif")
        
        # Vérifier que les MS normalisés existent
        if not os.path.exists(ms_norm_path):
            # Fallback sur MS non normalisé si MS_normalized n'existe pas
            if os.path.exists(ms_path):
                print(f"   ⚠️ {hour}: MS_normalized non trouvé, utilisation de MS")
                ms_norm_path = ms_path
            else:
                continue
        
        # === CHARGER MS NORMALISÉ ===
        with rasterio.open(ms_norm_path) as src:
            ms_data = src.read()  # Shape: (n_bands, height, width)
            ms_profile = src.profile.copy()
        
        n_bands = ms_data.shape[0]
        
        # Calculer NDVI à partir des données normalisées
        # NIR = bande 4 (index 4), Red = bande 2 (index 2)
        if n_bands >= 5:
            nir = ms_data[4].astype(np.float32)  # NIR
            red = ms_data[2].astype(np.float32)  # Red
            # Éviter division par zéro et valeurs invalides
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (nir - red) / (nir + red)
                ndvi = np.where(np.isfinite(ndvi), ndvi, -1)  # -1 pour les valeurs invalides (hors masque)
        else:
            ndvi = np.zeros_like(mask, dtype=np.float32)
        
        # === SEGMENTATION ===
        # Plantes = NDVI > seuil ET dans le masque
        segmentation = ((ndvi > NDVI_THRESHOLD) & (mask > 0)).astype(np.uint8)
        
        # Sauvegarder la segmentation
        if save_segmentation:
            seg_path = os.path.join(seg_dir, f"{plot_name}_{hour}_segmentation.tif")
            with rasterio.open(seg_path, 'w', driver='GTiff',
                              width=mask.shape[1], height=mask.shape[0],
                              count=1, dtype=np.uint8) as dst:
                dst.write(segmentation, 1)
        
        # === MASQUES ===
        mask_global = mask > 0  # Tout le polygone
        mask_plants = segmentation > 0  # Plantes seulement
        mask_non_plants = (mask > 0) & (segmentation == 0)  # Non-plantes dans le polygone
        
        n_global = np.sum(mask_global)
        n_plants = np.sum(mask_plants)
        n_non_plants = np.sum(mask_non_plants)
        
        result["n_pixels_global"] = int(n_global)
        result["n_pixels_plants"] = int(n_plants)
        result["n_pixels_non_plants"] = int(n_non_plants)
        result["plant_fraction"] = float(n_plants / n_global) if n_global > 0 else 0.0
        
        # === STATISTIQUES MS NORMALISÉ ===
        for band_idx in range(min(n_bands, len(MS_BAND_NAMES))):
            band_name = MS_BAND_NAMES[band_idx]
            band_data = ms_data[band_idx].astype(np.float32)
            
            # Global
            global_values = band_data[mask_global]
            valid_global = global_values[~np.isnan(global_values) & (global_values > 0)]
            result[f"{band_name}_global_mean"] = float(np.mean(valid_global)) if len(valid_global) > 0 else np.nan
            result[f"{band_name}_global_std"] = float(np.std(valid_global)) if len(valid_global) > 0 else np.nan
            
            # Plantes
            if n_plants > 0:
                plant_values = band_data[mask_plants]
                valid_plants = plant_values[~np.isnan(plant_values) & (plant_values > 0)]
                result[f"{band_name}_plants_mean"] = float(np.mean(valid_plants)) if len(valid_plants) > 0 else np.nan
                result[f"{band_name}_plants_std"] = float(np.std(valid_plants)) if len(valid_plants) > 0 else np.nan
            else:
                result[f"{band_name}_plants_mean"] = np.nan
                result[f"{band_name}_plants_std"] = np.nan
            
            # Non-plantes
            if n_non_plants > 0:
                non_plant_values = band_data[mask_non_plants]
                valid_non_plants = non_plant_values[~np.isnan(non_plant_values) & (non_plant_values > 0)]
                result[f"{band_name}_non_plants_mean"] = float(np.mean(valid_non_plants)) if len(valid_non_plants) > 0 else np.nan
                result[f"{band_name}_non_plants_std"] = float(np.std(valid_non_plants)) if len(valid_non_plants) > 0 else np.nan
            else:
                result[f"{band_name}_non_plants_mean"] = np.nan
                result[f"{band_name}_non_plants_std"] = np.nan
        
        # === STATISTIQUES TEMPÉRATURE ===
        if os.path.exists(temp_path):
            with rasterio.open(temp_path) as src:
                temp_data = src.read(1)
            
            # Global
            global_temp = temp_data[mask_global]
            valid_global_temp = global_temp[~np.isnan(global_temp) & (global_temp > 0)]
            result["Temp_global_mean"] = float(np.mean(valid_global_temp)) if len(valid_global_temp) > 0 else np.nan
            result["Temp_global_std"] = float(np.std(valid_global_temp)) if len(valid_global_temp) > 0 else np.nan
            
            # Plantes
            if n_plants > 0:
                plant_temp = temp_data[mask_plants]
                valid_plant_temp = plant_temp[~np.isnan(plant_temp) & (plant_temp > 0)]
                result["Temp_plants_mean"] = float(np.mean(valid_plant_temp)) if len(valid_plant_temp) > 0 else np.nan
                result["Temp_plants_std"] = float(np.std(valid_plant_temp)) if len(valid_plant_temp) > 0 else np.nan
            else:
                result["Temp_plants_mean"] = np.nan
                result["Temp_plants_std"] = np.nan
            
            # Non-plantes
            if n_non_plants > 0:
                non_plant_temp = temp_data[mask_non_plants]
                valid_non_plant_temp = non_plant_temp[~np.isnan(non_plant_temp) & (non_plant_temp > 0)]
                result["Temp_non_plants_mean"] = float(np.mean(valid_non_plant_temp)) if len(valid_non_plant_temp) > 0 else np.nan
                result["Temp_non_plants_std"] = float(np.std(valid_non_plant_temp)) if len(valid_non_plant_temp) > 0 else np.nan
            else:
                result["Temp_non_plants_mean"] = np.nan
                result["Temp_non_plants_std"] = np.nan
        
        # === NDVI ===
        # Global
        global_ndvi = ndvi[mask_global]
        valid_global_ndvi = global_ndvi[~np.isnan(global_ndvi)]
        result["NDVI_global_mean"] = float(np.mean(valid_global_ndvi)) if len(valid_global_ndvi) > 0 else np.nan
        result["NDVI_global_std"] = float(np.std(valid_global_ndvi)) if len(valid_global_ndvi) > 0 else np.nan
        
        # Plantes
        if n_plants > 0:
            plant_ndvi = ndvi[mask_plants]
            valid_plant_ndvi = plant_ndvi[~np.isnan(plant_ndvi)]
            result["NDVI_plants_mean"] = float(np.mean(valid_plant_ndvi)) if len(valid_plant_ndvi) > 0 else np.nan
            result["NDVI_plants_std"] = float(np.std(valid_plant_ndvi)) if len(valid_plant_ndvi) > 0 else np.nan
        else:
            result["NDVI_plants_mean"] = np.nan
            result["NDVI_plants_std"] = np.nan
        
        # Non-plantes
        if n_non_plants > 0:
            non_plant_ndvi = ndvi[mask_non_plants]
            valid_non_plant_ndvi = non_plant_ndvi[~np.isnan(non_plant_ndvi)]
            result["NDVI_non_plants_mean"] = float(np.mean(valid_non_plant_ndvi)) if len(valid_non_plant_ndvi) > 0 else np.nan
            result["NDVI_non_plants_std"] = float(np.std(valid_non_plant_ndvi)) if len(valid_non_plant_ndvi) > 0 else np.nan
        else:
            result["NDVI_non_plants_mean"] = np.nan
            result["NDVI_non_plants_std"] = np.nan
        
        results.append(result)
    
    return results


def save_plot_csv(plot_name: str, results: List[Dict]):
    """
    Sauvegarde les résultats d'un plot dans un CSV.
    """
    if not results:
        return None
    
    # Créer le dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, f"{plot_name}_analysis.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path


# =============================================================================
# MAIN
# =============================================================================

def analyze_single_plot(plot_name: str):
    """
    Analyse un seul plot.
    """
    print(f"\n{'='*60}")
    print(f"ANALYSE: {plot_name}")
    print(f"{'='*60}")
    
    results = analyze_plot(plot_name)
    
    if results:
        csv_path = save_plot_csv(plot_name, results)
        print(f"✅ CSV sauvegardé: {csv_path}")
        
        # Afficher un résumé
        df = pd.DataFrame(results)
        print("\nRésumé:")
        print(df[["hour", "n_pixels_global", "n_pixels_plants", "plant_fraction", 
                  "NDVI_global_mean", "Temp_global_mean"]].to_string())
    else:
        print("❌ Aucun résultat")


def analyze_all_plots(max_plots: int = None):
    """
    Analyse tous les plots.
    """
    print(f"\n{'='*60}")
    print("ANALYSE DE TOUS LES PLOTS")
    print("(utilise les MS normalisés calculés lors de l'extraction)")
    print(f"{'='*60}")
    
    # Lister tous les plots
    plot_dirs = [d for d in os.listdir(PATCHES_DIR) 
                 if os.path.isdir(os.path.join(PATCHES_DIR, d)) 
                 and (d.startswith("ETM_") or d.startswith("STR_"))]
    plot_dirs = sorted(plot_dirs)
    
    if max_plots:
        plot_dirs = plot_dirs[:max_plots]
        print(f"⚠️ Limité à {max_plots} plots (test)")
    
    print(f"📊 {len(plot_dirs)} plots à analyser")
    print()
    
    all_results = []
    t0 = time.time()
    
    for idx, plot_name in enumerate(plot_dirs):
        results = analyze_plot(plot_name, save_segmentation=True)
        
        if results:
            csv_path = save_plot_csv(plot_name, results)
            all_results.extend(results)
        
        # Progress
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"   {idx + 1}/{len(plot_dirs)} plots traités ({elapsed:.1f}s)")
    
    elapsed = time.time() - t0
    print(f"\n✅ {len(plot_dirs)} plots analysés en {elapsed:.1f}s")
    
    # Sauvegarder un CSV global
    if all_results:
        df = pd.DataFrame(all_results)
        global_csv = os.path.join(OUTPUT_DIR, "all_plots_analysis.csv")
        df.to_csv(global_csv, index=False)
        print(f"📄 CSV global: {global_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse des patches avec segmentation (utilise les MS normalisés)"
    )
    parser.add_argument("--plot", type=str, help="Nom du plot à analyser (ex: ETM_Rep1_001)")
    parser.add_argument("--all", action="store_true", help="Analyser tous les plots")
    parser.add_argument("--max-plots", type=int, help="Limite le nombre de plots (test)")
    
    args = parser.parse_args()
    
    if args.plot:
        analyze_single_plot(args.plot)
    elif args.all:
        analyze_all_plots(max_plots=args.max_plots)
    else:
        parser.print_help()
        print("\nExemples:")
        print("  python -m daily_sahel.analyze_results --plot ETM_Rep1_001")
        print("  python -m daily_sahel.analyze_results --all")
        print("  python -m daily_sahel.analyze_results --all --max-plots 10")


if __name__ == "__main__":
    main()
