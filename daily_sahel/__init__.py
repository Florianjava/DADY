"""
daily_sahel - Package pour l'analyse journalière des données drone au Sahel

Ce package permet d'extraire et analyser les données multispectrales et thermiques
acquises à différentes heures de la journée sur des parcelles expérimentales.

Modules:
    - extract_2021: Extraction des données pour l'année 2021
    - run_extraction_2021: Script principal d'extraction
    - analyze_results: Analyse et visualisation des résultats
    - verify_patches: Vérification et debug des patches exportés

Usage rapide:
    # Extraction complète
    python -m daily_sahel.run_extraction_2021
    
    # Analyse des résultats
    python -m daily_sahel.analyze_results
    
    # Vérification des patches
    python -m daily_sahel.verify_patches

Projet DADY - CIRAD
"""

__version__ = "0.1.0"
__author__ = "DADY Team"

from .extract_2021 import (
    extract_daily_data_2021,
    extract_patch_and_stats,
    excel_to_shp_plot,
    shp_to_excel_plot,
    compute_ndvi,
    get_file_paths,
    HOURS_2021,
    MS_BAND_NAMES,
    DATA_ROOT
)

from .analyze_results import (
    load_data,
    load_and_summarize,
    stats_by_hour,
    stats_by_genotype,
    daily_evolution_by_genotype,
    compare_genotypes,
    export_summary_csv
)

from .verify_patches import (
    check_directory_structure,
    check_patch,
    compare_patch_hours,
    verify_consistency,
    print_patch_report,
    list_patches_for_plot
)
