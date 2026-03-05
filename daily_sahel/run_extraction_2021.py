#!/usr/bin/env python3
"""
run_extraction_2021.py - Script principal d'extraction des données 2021

Ce script lance l'extraction complète des données journalières pour 2021,
avec export des patches canal par canal et génération du CSV récapitulatif.

Usage:
    python -m daily_sahel.run_extraction_2021
    
    ou directement:
    python daily_sahel/run_extraction_2021.py

Options configurables dans CONFIG ci-dessous.
"""

from .extract_2021 import extract_daily_data_2021

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Heures à traiter (None = toutes: 09h, 11h, 13h, 15h, 17h)
    "hours": None,
    
    # Chemin du CSV de sortie
    "output_csv": "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/daily_sahel_2021_complete.csv",
    
    # Répertoire pour les patches (None = pas d'export)
    "output_patches_dir": "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed",
    
    # Type de shapefile à utiliser ("STR" ou "ETM")
    "shapefile_type": "STR",
    
    # Afficher la progression
    "verbose": True,
}


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def main():
    """Lance l'extraction complète."""
    print("=" * 70)
    print("EXTRACTION DES DONNÉES JOURNALIÈRES 2021")
    print("=" * 70)
    print()
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    df = extract_daily_data_2021(
        hours=CONFIG["hours"],
        output_csv=CONFIG["output_csv"],
        output_patches_dir=CONFIG["output_patches_dir"],
        shapefile_type=CONFIG["shapefile_type"],
        verbose=CONFIG["verbose"]
    )
    
    print()
    print("=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Lignes extraites: {len(df)}")
    print(f"Heures: {df['hour'].unique().tolist()}")
    print(f"Plots: {df['plot_excel'].nunique()}")
    print(f"Génotypes: {df['geno'].nunique()}")
    
    return df


if __name__ == "__main__":
    main()
