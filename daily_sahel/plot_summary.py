"""
plot_summary.py - Génération de visualisations récapitulatives des résultats d'analyse

Ce script génère des graphiques moyennés sur tous les plots, en séparant ETM et STR :
1. Évolution temporelle des bandes MS (plantes vs non-plantes)
2. Évolution temporelle de la température
3. Évolution temporelle du NDVI
4. Fraction de végétation par heure
5. Comparaison ETM vs STR

Usage:
    python -m daily_sahel.plot_summary
    python -m daily_sahel.plot_summary --output-dir /path/to/output
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

# Configuration de matplotlib pour de belles figures
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoire des résultats d'analyse
ANALYSIS_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/analysis_results"
OUTPUT_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/analysis_plots"

# Heures
HOURS = ["09h", "11h", "13h", "15h", "17h"]
HOURS_NUMERIC = [9, 11, 13, 15, 17]

# Bandes MS
MS_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6"]

# Couleurs pour les graphiques
COLORS = {
    "ETM": "#1f77b4",  # Bleu
    "STR": "#ff7f0e",  # Orange
    "plants": "#2ca02c",  # Vert
    "non_plants": "#8c564b",  # Marron
    "global": "#7f7f7f",  # Gris
}

# Couleurs pour les bandes MS
BAND_COLORS = {
    "Blue": "#0000FF",
    "Green": "#00AA00",
    "Red": "#FF0000",
    "RedEdge": "#AA0055",
    "NIR": "#AA00AA",
    "Band6": "#555555",
}


# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================

def load_all_data(analysis_dir: str = ANALYSIS_DIR) -> pd.DataFrame:
    """
    Charge le CSV global avec tous les résultats.
    
    Returns:
        DataFrame avec toutes les données
    """
    csv_path = os.path.join(analysis_dir, "all_plots_analysis.csv")
    
    if not os.path.exists(csv_path):
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("   Lancez d'abord: python -m daily_sahel.analyze_results --all")
        return None
    
    df = pd.read_csv(csv_path)
    
    # Extraire le type (ETM ou STR) du nom du plot
    df["type"] = df["plot_name"].apply(lambda x: x.split("_")[0])
    
    # Extraire la répétition
    df["repetition"] = df["plot_name"].apply(lambda x: x.split("_")[1])
    
    # Convertir l'heure en numérique
    df["hour_num"] = df["hour"].apply(lambda x: int(x.replace("h", "")))
    
    print(f"✅ Données chargées: {len(df)} lignes")
    print(f"   - ETM: {len(df[df['type'] == 'ETM'])} lignes ({len(df[df['type'] == 'ETM']['plot_name'].unique())} plots)")
    print(f"   - STR: {len(df[df['type'] == 'STR'])} lignes ({len(df[df['type'] == 'STR']['plot_name'].unique())} plots)")
    
    return df


def compute_means_by_type_and_hour(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Calcule les moyennes et écarts-types par type (ETM/STR) et par heure.
    
    Args:
        df: DataFrame source
        columns: Colonnes à moyenner
    
    Returns:
        DataFrame avec mean et std par type et heure
    """
    # Grouper par type et heure
    grouped = df.groupby(["type", "hour_num"])
    
    # Calculer mean et std
    result = grouped[columns].agg(["mean", "std", "count"]).reset_index()
    
    # Aplatir les noms de colonnes
    result.columns = ["_".join(col).strip("_") for col in result.columns.values]
    
    return result


# =============================================================================
# GRAPHIQUES
# =============================================================================

def plot_temporal_evolution_by_band(df: pd.DataFrame, output_dir: str, region: str = "global"):
    """
    Graphique de l'évolution temporelle de chaque bande MS.
    Un subplot par bande, comparant ETM vs STR.
    
    Args:
        df: DataFrame source
        output_dir: Répertoire de sortie
        region: "global", "plants" ou "non_plants"
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    region_label = {
        "global": "Global (tout le polygone)",
        "plants": "Plantes (NDVI > 0)",
        "non_plants": "Non-plantes (sol)"
    }
    
    for idx, band in enumerate(MS_BAND_NAMES):
        ax = axes[idx]
        col_mean = f"{band}_{region}_mean"
        
        for plot_type in ["ETM", "STR"]:
            type_df = df[df["type"] == plot_type]
            
            # Grouper par heure
            grouped = type_df.groupby("hour_num")[col_mean].agg(["mean", "std", "count"])
            
            hours = grouped.index.values
            means = grouped["mean"].values
            stds = grouped["std"].values
            counts = grouped["count"].values
            
            # Erreur standard de la moyenne
            sem = stds / np.sqrt(counts)
            
            # Plot avec bande d'erreur
            ax.plot(hours, means, 'o-', color=COLORS[plot_type], label=plot_type, linewidth=2, markersize=6)
            ax.fill_between(hours, means - sem, means + sem, color=COLORS[plot_type], alpha=0.2)
        
        ax.set_title(band, fontweight='bold')
        ax.set_xlabel("Heure")
        ax.set_ylabel("Réflectance normalisée")
        ax.set_xticks(HOURS_NUMERIC)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Évolution temporelle des bandes MS - {region_label[region]}", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"temporal_MS_bands_{region}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_ndvi_temporal(df: pd.DataFrame, output_dir: str):
    """
    Graphique de l'évolution temporelle du NDVI.
    Plantes vs Non-plantes, ETM vs STR.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    regions = ["global", "plants", "non_plants"]
    region_labels = {
        "global": "Global",
        "plants": "Plantes (NDVI > 0)",
        "non_plants": "Sol (NDVI ≤ 0)"
    }
    
    for ax, region in zip(axes, regions):
        col_mean = f"NDVI_{region}_mean"
        
        for plot_type in ["ETM", "STR"]:
            type_df = df[df["type"] == plot_type]
            
            grouped = type_df.groupby("hour_num")[col_mean].agg(["mean", "std", "count"])
            
            hours = grouped.index.values
            means = grouped["mean"].values
            stds = grouped["std"].values
            counts = grouped["count"].values
            sem = stds / np.sqrt(counts)
            
            ax.plot(hours, means, 'o-', color=COLORS[plot_type], label=plot_type, linewidth=2, markersize=8)
            ax.fill_between(hours, means - sem, means + sem, color=COLORS[plot_type], alpha=0.2)
        
        ax.set_title(region_labels[region], fontweight='bold')
        ax.set_xlabel("Heure")
        ax.set_ylabel("NDVI")
        ax.set_xticks(HOURS_NUMERIC)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Évolution temporelle du NDVI", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "temporal_NDVI.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_temperature_temporal(df: pd.DataFrame, output_dir: str):
    """
    Graphique de l'évolution temporelle de la température.
    Plantes vs Non-plantes vs Global, ETM vs STR.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, plot_type in zip(axes, ["ETM", "STR"]):
        type_df = df[df["type"] == plot_type]
        
        for region, linestyle in [("global", "-"), ("plants", "--"), ("non_plants", ":")]:
            col_mean = f"Temp_{region}_mean"
            
            grouped = type_df.groupby("hour_num")[col_mean].agg(["mean", "std", "count"])
            
            hours = grouped.index.values
            means = grouped["mean"].values
            stds = grouped["std"].values
            counts = grouped["count"].values
            sem = stds / np.sqrt(counts)
            
            region_colors = {"global": "#7f7f7f", "plants": "#2ca02c", "non_plants": "#8c564b"}
            region_labels = {"global": "Global", "plants": "Plantes", "non_plants": "Sol"}
            
            ax.plot(hours, means, 'o' + linestyle, color=region_colors[region], 
                   label=region_labels[region], linewidth=2, markersize=6)
            ax.fill_between(hours, means - sem, means + sem, color=region_colors[region], alpha=0.15)
        
        ax.set_title(f"{plot_type}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Heure")
        ax.set_ylabel("Température (°C)")
        ax.set_xticks(HOURS_NUMERIC)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Évolution temporelle de la température", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "temporal_Temperature.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_plant_fraction(df: pd.DataFrame, output_dir: str):
    """
    Graphique de la fraction de végétation par heure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for plot_type in ["ETM", "STR"]:
        type_df = df[df["type"] == plot_type]
        
        grouped = type_df.groupby("hour_num")["plant_fraction"].agg(["mean", "std", "count"])
        
        hours = grouped.index.values
        means = grouped["mean"].values * 100  # Convertir en pourcentage
        stds = grouped["std"].values * 100
        counts = grouped["count"].values
        sem = stds / np.sqrt(counts)
        
        ax.plot(hours, means, 'o-', color=COLORS[plot_type], label=plot_type, linewidth=2, markersize=8)
        ax.fill_between(hours, means - sem, means + sem, color=COLORS[plot_type], alpha=0.2)
    
    ax.set_title("Fraction de végétation détectée (NDVI > 0)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Heure")
    ax.set_ylabel("Fraction de végétation (%)")
    ax.set_xticks(HOURS_NUMERIC)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "plant_fraction.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_comparison_etm_str_boxplot(df: pd.DataFrame, output_dir: str):
    """
    Boxplots comparant ETM vs STR pour différentes métriques par heure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ("NDVI_plants_mean", "NDVI Plantes"),
        ("NIR_plants_mean", "NIR Plantes"),
        ("Red_plants_mean", "Red Plantes"),
        ("Temp_plants_mean", "Température Plantes (°C)"),
        ("plant_fraction", "Fraction végétation"),
        ("NDVI_global_mean", "NDVI Global"),
    ]
    
    for ax, (col, label) in zip(axes.flatten(), metrics):
        data_etm = []
        data_str = []
        positions_etm = []
        positions_str = []
        
        for i, hour in enumerate(HOURS_NUMERIC):
            hour_df = df[df["hour_num"] == hour]
            etm_values = hour_df[hour_df["type"] == "ETM"][col].dropna().values
            str_values = hour_df[hour_df["type"] == "STR"][col].dropna().values
            
            if len(etm_values) > 0:
                data_etm.append(etm_values)
                positions_etm.append(i - 0.15)
            if len(str_values) > 0:
                data_str.append(str_values)
                positions_str.append(i + 0.15)
        
        # Boxplots
        if data_etm:
            bp_etm = ax.boxplot(data_etm, positions=positions_etm, widths=0.25, patch_artist=True,
                               showfliers=False)
            for patch in bp_etm['boxes']:
                patch.set_facecolor(COLORS["ETM"])
                patch.set_alpha(0.6)
        
        if data_str:
            bp_str = ax.boxplot(data_str, positions=positions_str, widths=0.25, patch_artist=True,
                               showfliers=False)
            for patch in bp_str['boxes']:
                patch.set_facecolor(COLORS["STR"])
                patch.set_alpha(0.6)
        
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(range(len(HOURS_NUMERIC)))
        ax.set_xticklabels([f"{h}h" for h in HOURS_NUMERIC])
        ax.set_xlabel("Heure")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Légende
        etm_patch = mpatches.Patch(color=COLORS["ETM"], alpha=0.6, label='ETM')
        str_patch = mpatches.Patch(color=COLORS["STR"], alpha=0.6, label='STR')
        ax.legend(handles=[etm_patch, str_patch], loc='upper right')
    
    plt.suptitle("Comparaison ETM vs STR par heure", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "comparison_ETM_STR_boxplot.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_spectral_signature(df: pd.DataFrame, output_dir: str):
    """
    Signature spectrale moyenne des plantes vs sol, par heure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Un subplot par heure + un pour la moyenne globale
    for idx, hour in enumerate(HOURS_NUMERIC):
        if idx < 5:
            ax = axes.flatten()[idx]
        else:
            continue
        
        hour_df = df[df["hour_num"] == hour]
        
        for plot_type in ["ETM", "STR"]:
            type_df = hour_df[hour_df["type"] == plot_type]
            
            for region, linestyle in [("plants", "-"), ("non_plants", "--")]:
                means = []
                stds = []
                
                for band in MS_BAND_NAMES:
                    col = f"{band}_{region}_mean"
                    val = type_df[col].mean()
                    std = type_df[col].std() / np.sqrt(len(type_df))
                    means.append(val)
                    stds.append(std)
                
                color = COLORS[plot_type]
                label = f"{plot_type} - {'Plantes' if region == 'plants' else 'Sol'}"
                ax.plot(MS_BAND_NAMES, means, linestyle, color=color, label=label, 
                       linewidth=2, marker='o', markersize=5)
        
        ax.set_title(f"{hour}h", fontweight='bold')
        ax.set_xlabel("Bande spectrale")
        ax.set_ylabel("Réflectance normalisée")
        ax.tick_params(axis='x', rotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Dernier subplot : moyenne sur toutes les heures
    ax = axes.flatten()[5]
    
    for plot_type in ["ETM", "STR"]:
        type_df = df[df["type"] == plot_type]
        
        for region, linestyle in [("plants", "-"), ("non_plants", "--")]:
            means = []
            
            for band in MS_BAND_NAMES:
                col = f"{band}_{region}_mean"
                means.append(type_df[col].mean())
            
            color = COLORS[plot_type]
            label = f"{plot_type} - {'Plantes' if region == 'plants' else 'Sol'}"
            ax.plot(MS_BAND_NAMES, means, linestyle, color=color, label=label, 
                   linewidth=2, marker='o', markersize=5)
    
    ax.set_title("Moyenne toutes heures", fontweight='bold')
    ax.set_xlabel("Bande spectrale")
    ax.set_ylabel("Réflectance normalisée")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("Signature spectrale Plantes vs Sol", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "spectral_signature.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_difference_plants_soil(df: pd.DataFrame, output_dir: str):
    """
    Différence entre plantes et sol (∆ = Plantes - Sol) par bande et par heure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, band in enumerate(MS_BAND_NAMES):
        ax = axes[idx]
        
        for plot_type in ["ETM", "STR"]:
            type_df = df[df["type"] == plot_type]
            
            deltas_mean = []
            deltas_sem = []
            
            for hour in HOURS_NUMERIC:
                hour_df = type_df[type_df["hour_num"] == hour]
                
                plants_col = f"{band}_plants_mean"
                soil_col = f"{band}_non_plants_mean"
                
                # Différence par plot
                delta = hour_df[plants_col] - hour_df[soil_col]
                delta = delta.dropna()
                
                if len(delta) > 0:
                    deltas_mean.append(delta.mean())
                    deltas_sem.append(delta.std() / np.sqrt(len(delta)))
                else:
                    deltas_mean.append(np.nan)
                    deltas_sem.append(0)
            
            deltas_mean = np.array(deltas_mean)
            deltas_sem = np.array(deltas_sem)
            
            ax.plot(HOURS_NUMERIC, deltas_mean, 'o-', color=COLORS[plot_type], 
                   label=plot_type, linewidth=2, markersize=6)
            ax.fill_between(HOURS_NUMERIC, deltas_mean - deltas_sem, deltas_mean + deltas_sem,
                           color=COLORS[plot_type], alpha=0.2)
        
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_title(f"∆{band} (Plantes - Sol)", fontweight='bold')
        ax.set_xlabel("Heure")
        ax.set_ylabel("Différence")
        ax.set_xticks(HOURS_NUMERIC)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("Contraste Plantes vs Sol par bande spectrale", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "delta_plants_soil.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_heatmap_summary(df: pd.DataFrame, output_dir: str):
    """
    Heatmap récapitulatif des valeurs moyennes par heure et type.
    """
    metrics = ["NDVI_plants_mean", "NDVI_global_mean", "Temp_plants_mean", 
               "NIR_plants_mean", "Red_plants_mean", "plant_fraction"]
    metric_labels = ["NDVI Plantes", "NDVI Global", "Temp Plantes", 
                     "NIR Plantes", "Red Plantes", "Fraction vég."]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    
    for ax, plot_type in zip(axes, ["ETM", "STR"]):
        type_df = df[df["type"] == plot_type]
        
        # Créer la matrice
        matrix = []
        for metric in metrics:
            row = []
            for hour in HOURS_NUMERIC:
                hour_df = type_df[type_df["hour_num"] == hour]
                val = hour_df[metric].mean()
                row.append(val)
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Normaliser chaque ligne (pour avoir une échelle comparable)
        matrix_norm = np.zeros_like(matrix)
        for i in range(matrix.shape[0]):
            row_min = np.nanmin(matrix[i])
            row_max = np.nanmax(matrix[i])
            if row_max > row_min:
                matrix_norm[i] = (matrix[i] - row_min) / (row_max - row_min)
            else:
                matrix_norm[i] = 0.5
        
        # Heatmap
        im = ax.imshow(matrix_norm, cmap='viridis', aspect='auto')
        
        # Labels
        ax.set_xticks(range(len(HOURS_NUMERIC)))
        ax.set_xticklabels([f"{h}h" for h in HOURS_NUMERIC])
        ax.set_yticks(range(len(metric_labels)))
        ax.set_yticklabels(metric_labels)
        
        # Annoter avec les valeurs réelles
        for i in range(len(metric_labels)):
            for j in range(len(HOURS_NUMERIC)):
                val = matrix[i, j]
                if metric_labels[i] == "Fraction vég.":
                    text = f"{val*100:.1f}%"
                elif "Temp" in metric_labels[i]:
                    text = f"{val:.1f}"
                elif "NDVI" in metric_labels[i]:
                    text = f"{val:.3f}"
                else:
                    text = f"{val:.0f}"
                ax.text(j, i, text, ha='center', va='center', fontsize=9, 
                       color='white' if matrix_norm[i, j] < 0.5 else 'black')
        
        ax.set_title(f"{plot_type}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Heure")
    
    plt.suptitle("Récapitulatif des métriques par heure (valeurs normalisées par ligne)", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "heatmap_summary.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def plot_by_repetition(df: pd.DataFrame, output_dir: str):
    """
    Graphiques par répétition (Rep1, Rep2, Rep3, Rep4) pour voir la variabilité.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = [
        ("NDVI_plants_mean", "NDVI Plantes"),
        ("Temp_plants_mean", "Température Plantes (°C)"),
        ("NIR_plants_mean", "NIR Plantes"),
        ("plant_fraction", "Fraction végétation"),
    ]
    
    repetitions = ["Rep1", "Rep2", "Rep3", "Rep4"]
    rep_colors = plt.cm.tab10(np.linspace(0, 0.4, 4))
    
    for ax, (col, label) in zip(axes, metrics):
        for plot_type in ["ETM", "STR"]:
            type_df = df[df["type"] == plot_type]
            linestyle = '-' if plot_type == "ETM" else '--'
            
            for rep_idx, rep in enumerate(repetitions):
                rep_df = type_df[type_df["repetition"] == rep]
                
                grouped = rep_df.groupby("hour_num")[col].agg(["mean", "std", "count"])
                hours = grouped.index.values
                means = grouped["mean"].values
                
                # Multiplier fraction par 100
                if col == "plant_fraction":
                    means = means * 100
                
                alpha = 1.0 if plot_type == "ETM" else 0.7
                ax.plot(hours, means, linestyle, color=rep_colors[rep_idx], 
                       label=f"{plot_type} {rep}" if plot_type == "ETM" else None,
                       linewidth=2, alpha=alpha)
        
        ax.set_title(label, fontweight='bold')
        ax.set_xlabel("Heure")
        ax.set_ylabel(label.split('(')[0])
        ax.set_xticks(HOURS_NUMERIC)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Ajouter une légende pour expliquer les lignes pleines vs pointillées
    fig.text(0.5, -0.02, "Ligne pleine = ETM, Ligne pointillée = STR", 
             ha='center', fontsize=11, style='italic')
    
    plt.suptitle("Évolution par répétition", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "by_repetition.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"   📊 {output_path}")


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """
    Crée un tableau récapitulatif des moyennes par type et heure.
    """
    summary_data = []
    
    for plot_type in ["ETM", "STR"]:
        type_df = df[df["type"] == plot_type]
        
        for hour in HOURS_NUMERIC:
            hour_df = type_df[type_df["hour_num"] == hour]
            
            row = {
                "Type": plot_type,
                "Heure": f"{hour}h",
                "N_plots": len(hour_df["plot_name"].unique()),
                "NDVI_global": hour_df["NDVI_global_mean"].mean(),
                "NDVI_plants": hour_df["NDVI_plants_mean"].mean(),
                "Temp_plants": hour_df["Temp_plants_mean"].mean(),
                "Temp_sol": hour_df["Temp_non_plants_mean"].mean(),
                "Fraction_veg_%": hour_df["plant_fraction"].mean() * 100,
                "NIR_plants": hour_df["NIR_plants_mean"].mean(),
                "Red_plants": hour_df["Red_plants_mean"].mean(),
            }
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Arrondir les valeurs
    for col in summary_df.columns:
        if col not in ["Type", "Heure", "N_plots"]:
            if "NDVI" in col:
                summary_df[col] = summary_df[col].round(4)
            elif "Temp" in col:
                summary_df[col] = summary_df[col].round(2)
            elif "Fraction" in col:
                summary_df[col] = summary_df[col].round(2)
            else:
                summary_df[col] = summary_df[col].round(1)
    
    # Sauvegarder en CSV
    csv_path = os.path.join(output_dir, "summary_table.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"   📄 {csv_path}")
    
    return summary_df


# =============================================================================
# MAIN
# =============================================================================

def generate_all_plots(analysis_dir: str = ANALYSIS_DIR, output_dir: str = OUTPUT_DIR):
    """
    Génère tous les graphiques récapitulatifs.
    """
    print(f"\n{'='*60}")
    print("GÉNÉRATION DES VISUALISATIONS RÉCAPITULATIVES")
    print(f"{'='*60}")
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Dossier de sortie: {output_dir}\n")
    
    # Charger les données
    df = load_all_data(analysis_dir)
    if df is None:
        return
    
    print(f"\n📊 Génération des graphiques...\n")
    
    # 1. Évolution temporelle des bandes MS
    print("1. Bandes MS - Global")
    plot_temporal_evolution_by_band(df, output_dir, region="global")
    print("2. Bandes MS - Plantes")
    plot_temporal_evolution_by_band(df, output_dir, region="plants")
    print("3. Bandes MS - Sol")
    plot_temporal_evolution_by_band(df, output_dir, region="non_plants")
    
    # 2. NDVI
    print("4. NDVI temporal")
    plot_ndvi_temporal(df, output_dir)
    
    # 3. Température
    print("5. Température temporelle")
    plot_temperature_temporal(df, output_dir)
    
    # 4. Fraction végétation
    print("6. Fraction de végétation")
    plot_plant_fraction(df, output_dir)
    
    # 5. Comparaison ETM vs STR boxplot
    print("7. Comparaison ETM vs STR (boxplots)")
    plot_comparison_etm_str_boxplot(df, output_dir)
    
    # 6. Signature spectrale
    print("8. Signature spectrale")
    plot_spectral_signature(df, output_dir)
    
    # 7. Différence plantes-sol
    print("9. Contraste plantes vs sol")
    plot_difference_plants_soil(df, output_dir)
    
    # 8. Heatmap
    print("10. Heatmap récapitulatif")
    plot_heatmap_summary(df, output_dir)
    
    # 9. Par répétition
    print("11. Par répétition")
    plot_by_repetition(df, output_dir)
    
    # 10. Tableau récapitulatif
    print("12. Tableau récapitulatif")
    summary_df = create_summary_table(df, output_dir)
    
    print(f"\n{'='*60}")
    print("✅ TERMINÉ!")
    print(f"{'='*60}")
    print(f"\n📁 Tous les graphiques sont dans: {output_dir}")
    print(f"\n📄 Tableau récapitulatif:")
    print(summary_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Génération de visualisations récapitulatives des résultats d'analyse"
    )
    parser.add_argument("--analysis-dir", type=str, default=ANALYSIS_DIR,
                       help="Répertoire contenant les résultats d'analyse")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                       help="Répertoire de sortie pour les graphiques")
    
    args = parser.parse_args()
    
    generate_all_plots(analysis_dir=args.analysis_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
