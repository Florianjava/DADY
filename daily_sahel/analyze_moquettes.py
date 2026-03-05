"""
analyze_moquettes.py - Analyse des moquettes de référence au cours de la journée

Ce script extrait les valeurs des canaux MS pour les 3 moquettes de référence
à chaque heure de la journée, et génère des visualisations pour comprendre
les variations temporelles.

Les moquettes sont des zones de référence stables qui permettent de normaliser
les images entre les différentes heures d'acquisition.

Usage:
    python -m daily_sahel.analyze_moquettes
"""

import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoire des images alignées
ALIGNED_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration/aligned_2021"

# Répertoire de sortie
OUTPUT_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/moquettes_analysis"

# Heures disponibles
HOURS = ["09h", "11h", "13h", "15h", "17h"]

# Coordonnées des moquettes (x, y) dans l'espace image aligné
# x = colonne, y = ligne
MOQUETTES = {
    "Moquette_1": {"x": 3230, "y": 964},
    "Moquette_2": {"x": 3161, "y": 1326},
    "Moquette_3": {"x": 3111, "y": 1600},
}

# Taille du carré d'extraction (en pixels)
PATCH_SIZE = 25

# Noms des bandes MS
MS_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6"]

# Couleurs "naturelles" pour chaque bande
BAND_COLORS = {
    "Blue": "#9370DB",      # Violet clair (Medium Purple)
    "Green": "#228B22",     # Vert forêt
    "Red": "#DC143C",       # Rouge crimson
    "RedEdge": "#8B0000",   # Rouge sombre (Dark Red)
    "NIR": "#FF6347",       # Tomato (proche infrarouge - pas visible, on met orange-rouge)
    "Band6": "#708090",     # Gris ardoise
}

# Épaisseurs de ligne pour chaque moquette
MOQUETTE_LINEWIDTHS = {
    "Moquette_1": 3.0,
    "Moquette_2": 2.0,
    "Moquette_3": 1.0,
}

# Styles de ligne pour chaque moquette
MOQUETTE_LINESTYLES = {
    "Moquette_1": "-",      # Solide
    "Moquette_2": "--",     # Tirets
    "Moquette_3": ":",      # Pointillés
}


# =============================================================================
# EXTRACTION DES DONNÉES
# =============================================================================

def extract_moquette_values(hour: str) -> dict:
    """
    Extrait les valeurs moyennes des canaux MS pour chaque moquette à une heure donnée.
    
    Args:
        hour: Heure ("09h", "11h", etc.)
    
    Returns:
        Dict avec les valeurs par moquette et par bande
    """
    ms_path = os.path.join(ALIGNED_DIR, f"{hour}_MS_aligned.tif")
    
    if not os.path.exists(ms_path):
        print(f"⚠️ Fichier non trouvé: {ms_path}")
        return None
    
    results = {"hour": hour}
    
    with rasterio.open(ms_path) as src:
        # Lire toutes les bandes
        ms_data = src.read()  # Shape: (n_bands, height, width)
        n_bands = ms_data.shape[0]
        
        for moquette_name, coords in MOQUETTES.items():
            x, y = coords["x"], coords["y"]
            half_size = PATCH_SIZE // 2
            
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
                
                # Exclure les valeurs nulles/NaN
                valid_values = band_patch[band_patch > 0]
                
                if len(valid_values) > 0:
                    mean_val = float(np.mean(valid_values))
                    std_val = float(np.std(valid_values))
                else:
                    mean_val = np.nan
                    std_val = np.nan
                
                results[f"{moquette_name}_{band_name}_mean"] = mean_val
                results[f"{moquette_name}_{band_name}_std"] = std_val
    
    return results


def extract_all_hours() -> pd.DataFrame:
    """
    Extrait les valeurs pour toutes les heures et retourne un DataFrame.
    """
    all_results = []
    
    for hour in HOURS:
        print(f"📊 Extraction {hour}...")
        result = extract_moquette_values(hour)
        if result:
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    return df


# =============================================================================
# VISUALISATIONS
# =============================================================================

def plot_bands_by_moquette(df: pd.DataFrame, output_dir: str):
    """
    Crée un plot par moquette montrant l'évolution de chaque bande.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    hours_numeric = [9, 11, 13, 15, 17]
    
    for idx, moquette_name in enumerate(MOQUETTES.keys()):
        ax = axes[idx]
        
        for band_name in MS_BAND_NAMES:
            col_name = f"{moquette_name}_{band_name}_mean"
            if col_name in df.columns:
                values = df[col_name].values
                color = BAND_COLORS[band_name]
                ax.plot(hours_numeric, values, 
                       color=color, 
                       linewidth=2.5,
                       marker='o',
                       markersize=8,
                       label=band_name)
        
        ax.set_xlabel("Heure", fontsize=12)
        ax.set_ylabel("Valeur moyenne (DN)", fontsize=12)
        ax.set_title(f"{moquette_name}", fontsize=14, fontweight='bold')
        ax.set_xticks(hours_numeric)
        ax.set_xticklabels(HOURS)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Évolution des canaux MS par moquette au cours de la journée", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "moquettes_bands_by_moquette.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {out_path}")
    plt.close()


def plot_moquettes_by_band(df: pd.DataFrame, output_dir: str):
    """
    Crée un plot par bande montrant l'évolution de chaque moquette.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    hours_numeric = [9, 11, 13, 15, 17]
    
    for idx, band_name in enumerate(MS_BAND_NAMES):
        ax = axes[idx]
        color = BAND_COLORS[band_name]
        
        for moquette_name in MOQUETTES.keys():
            col_name = f"{moquette_name}_{band_name}_mean"
            if col_name in df.columns:
                values = df[col_name].values
                lw = MOQUETTE_LINEWIDTHS[moquette_name]
                ls = MOQUETTE_LINESTYLES[moquette_name]
                ax.plot(hours_numeric, values, 
                       color=color, 
                       linewidth=lw,
                       linestyle=ls,
                       marker='o',
                       markersize=8,
                       label=moquette_name)
        
        ax.set_xlabel("Heure", fontsize=12)
        ax.set_ylabel("Valeur moyenne (DN)", fontsize=12)
        ax.set_title(f"{band_name}", fontsize=14, fontweight='bold', color=color)
        ax.set_xticks(hours_numeric)
        ax.set_xticklabels(HOURS)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Évolution des moquettes par canal MS au cours de la journée\n(trait épais=Moq1, tirets=Moq2, pointillés=Moq3)", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "moquettes_moquettes_by_band.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {out_path}")
    plt.close()


def plot_ndvi_comparison(df: pd.DataFrame, output_dir: str):
    """
    Crée un plot comparant le NDVI calculé pour chaque moquette.
    NDVI = (NIR - Red) / (NIR + Red)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    hours_numeric = [9, 11, 13, 15, 17]
    
    for moquette_name in MOQUETTES.keys():
        nir_col = f"{moquette_name}_NIR_mean"
        red_col = f"{moquette_name}_Red_mean"
        
        if nir_col in df.columns and red_col in df.columns:
            nir = df[nir_col].values
            red = df[red_col].values
            ndvi = (nir - red) / (nir + red + 1e-10)
            
            lw = MOQUETTE_LINEWIDTHS[moquette_name]
            ls = MOQUETTE_LINESTYLES[moquette_name]
            ax.plot(hours_numeric, ndvi, 
                   linewidth=lw,
                   linestyle=ls,
                   marker='o',
                   markersize=10,
                   label=moquette_name)
    
    ax.set_xlabel("Heure", fontsize=12)
    ax.set_ylabel("NDVI", fontsize=12)
    ax.set_title("Évolution du NDVI des moquettes au cours de la journée", 
                fontsize=14, fontweight='bold')
    ax.set_xticks(hours_numeric)
    ax.set_xticklabels(HOURS)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, "moquettes_ndvi_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {out_path}")
    plt.close()


def plot_all_bands_all_moquettes(df: pd.DataFrame, output_dir: str):
    """
    Crée un plot unique avec toutes les bandes et toutes les moquettes.
    Différenciation par couleur (bande) et épaisseur/style (moquette).
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    hours_numeric = [9, 11, 13, 15, 17]
    
    for band_name in MS_BAND_NAMES:
        color = BAND_COLORS[band_name]
        
        for moquette_name in MOQUETTES.keys():
            col_name = f"{moquette_name}_{band_name}_mean"
            if col_name in df.columns:
                values = df[col_name].values
                lw = MOQUETTE_LINEWIDTHS[moquette_name]
                ls = MOQUETTE_LINESTYLES[moquette_name]
                
                # Label seulement pour la première moquette de chaque bande
                label = f"{band_name}" if moquette_name == "Moquette_1" else None
                
                ax.plot(hours_numeric, values, 
                       color=color, 
                       linewidth=lw,
                       linestyle=ls,
                       marker='o',
                       markersize=6,
                       label=label,
                       alpha=0.8)
    
    ax.set_xlabel("Heure", fontsize=12)
    ax.set_ylabel("Valeur moyenne (DN)", fontsize=12)
    ax.set_title("Évolution de tous les canaux pour toutes les moquettes\n(trait épais=Moq1, tirets=Moq2, pointillés=Moq3)", 
                fontsize=14, fontweight='bold')
    ax.set_xticks(hours_numeric)
    ax.set_xticklabels(HOURS)
    ax.legend(loc='best', fontsize=11, title="Canaux")
    ax.grid(True, alpha=0.3)
    
    out_path = os.path.join(output_dir, "moquettes_all_bands_all_moquettes.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {out_path}")
    plt.close()


def plot_normalized_bands(df: pd.DataFrame, output_dir: str):
    """
    Crée un plot avec les valeurs normalisées (ratio par rapport à 13h)
    pour mieux voir les variations relatives.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    hours_numeric = [9, 11, 13, 15, 17]
    ref_hour_idx = 2  # 13h est l'index 2
    
    for idx, moquette_name in enumerate(MOQUETTES.keys()):
        ax = axes[idx]
        
        for band_name in MS_BAND_NAMES:
            col_name = f"{moquette_name}_{band_name}_mean"
            if col_name in df.columns:
                values = df[col_name].values
                ref_value = values[ref_hour_idx]
                
                if ref_value > 0:
                    normalized = values / ref_value
                else:
                    normalized = values
                
                color = BAND_COLORS[band_name]
                ax.plot(hours_numeric, normalized, 
                       color=color, 
                       linewidth=2.5,
                       marker='o',
                       markersize=8,
                       label=band_name)
        
        ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel("Heure", fontsize=12)
        ax.set_ylabel("Ratio (/ valeur 13h)", fontsize=12)
        ax.set_title(f"{moquette_name}", fontsize=14, fontweight='bold')
        ax.set_xticks(hours_numeric)
        ax.set_xticklabels(HOURS)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Évolution normalisée des canaux (référence = 13h)", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, "moquettes_normalized_bands.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"✅ Sauvegardé: {out_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ANALYSE DES MOQUETTES DE RÉFÉRENCE")
    print("=" * 60)
    print()
    
    # Créer le répertoire de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Extraire les données
    print("📊 Extraction des valeurs des moquettes...")
    df = extract_all_hours()
    
    if df.empty:
        print("❌ Aucune donnée extraite!")
        return
    
    # Sauvegarder le CSV
    csv_path = os.path.join(OUTPUT_DIR, "moquettes_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV sauvegardé: {csv_path}")
    print()
    
    # Afficher un résumé
    print("📋 Résumé des données:")
    print(df.to_string())
    print()
    
    # Générer les visualisations
    print("📈 Génération des visualisations...")
    plot_bands_by_moquette(df, OUTPUT_DIR)
    plot_moquettes_by_band(df, OUTPUT_DIR)
    plot_ndvi_comparison(df, OUTPUT_DIR)
    plot_all_bands_all_moquettes(df, OUTPUT_DIR)
    plot_normalized_bands(df, OUTPUT_DIR)
    
    print()
    print("=" * 60)
    print("✅ ANALYSE TERMINÉE")
    print(f"   Résultats dans: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
