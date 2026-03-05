#!/usr/bin/env python3
"""
prepare_for_fijiyama.py - Prépare les orthomosaïques pour le recalage dans Fijiyama

Ce script exporte pour chaque heure les ORTHOMOSAÏQUES COMPLÈTES :
    - mean_MS.tif : moyenne de tous les canaux multispectraux (6 bandes)
    - NDVI.tif : indice de végétation (NIR - Red) / (NIR + Red)
    - Temp.tif : image thermique

Ces fichiers peuvent ensuite être ouverts dans Fiji/Fijiyama pour calculer
les transformations de recalage entre les différentes heures.

IMPORTANT: On travaille sur les orthomosaïques complètes, pas sur les patches !
Le recalage doit être calculé sur l'ensemble de l'image pour avoir des 
transformations valides.

Usage:
    python -m registration.prepare_for_fijiyama
    
    ou en important:
    from registration.prepare_for_fijiyama import prepare_orthomosaics_for_fijiyama
"""

import os
import numpy as np
import rasterio
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import shutil


# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoire source des orthomosaïques
SOURCE_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/anatomics"

# Répertoire de sortie pour Fijiyama
OUTPUT_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"

# Heures disponibles pour 2021
HOURS_2021 = ["09h", "11h", "13h", "15h", "17h"]

# Heure de référence pour le recalage
REFERENCE_HOUR = "13h"

# Indices des bandes dans l'orthomosaïque MS (6 bandes)
# Blue=0, Green=1, Red=2, RedEdge=3, NIR=4, Band6=5
BAND_INDICES = {
    "Blue": 0,
    "Green": 1,
    "Red": 2,
    "RedEdge": 3,
    "NIR": 4,
    "Band6": 5
}


# =============================================================================
# FONCTIONS POUR LES ORTHOMOSAÏQUES
# =============================================================================

def get_orthomosaic_paths(year: str = "2021", hour: str = "13h", 
                          source_dir: str = SOURCE_DIR) -> Dict[str, str]:
    """
    Retourne les chemins des orthomosaïques MS et IR pour une heure donnée.
    
    Args:
        year: Année ("2021" ou "2022")
        hour: Heure ("09h", "11h", "13h", "15h", "17h")
        source_dir: Répertoire source
        
    Returns:
        Dict avec les chemins MS et IR
    """
    # Format des noms de fichiers : 2021_09h_MS.tif, 2021_09h_IR.tif
    # Note: 09h est noté "9h" dans certains fichiers
    hour_variants = [hour, hour.lstrip("0")]  # "09h" -> ["09h", "9h"]
    
    paths = {"MS": None, "IR": None}
    
    for h in hour_variants:
        ms_path = os.path.join(source_dir, f"{year}_{h}_MS.tif")
        ir_path = os.path.join(source_dir, f"{year}_{h}_IR.tif")
        
        if os.path.exists(ms_path):
            paths["MS"] = ms_path
        if os.path.exists(ir_path):
            paths["IR"] = ir_path
            
    return paths


def compute_mean_ms(ms_path: str) -> Tuple[np.ndarray, dict]:
    """
    Calcule la moyenne des 6 bandes multispectrales de l'orthomosaïque.
    
    Args:
        ms_path: Chemin de l'orthomosaïque MS
        
    Returns:
        (mean_data, profile) : données moyennées et profil rasterio
    """
    with rasterio.open(ms_path) as src:
        # Lire toutes les bandes
        data = src.read().astype(np.float32)
        profile = src.profile.copy()
    
    # Moyenne sur l'axe des bandes (axis=0)
    mean_data = np.mean(data, axis=0)
    
    # Mettre à jour le profil pour une seule bande
    profile.update(dtype=np.float32, count=1)
    
    return mean_data, profile


def compute_ndvi_from_bands(ms_path: str) -> Tuple[np.ndarray, dict]:
    """
    Calcule le NDVI à partir des bandes Red et NIR de l'orthomosaïque.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        ms_path: Chemin de l'orthomosaïque MS
        
    Returns:
        (ndvi_data, profile) : NDVI et profil rasterio
    """
    with rasterio.open(ms_path) as src:
        red = src.read(BAND_INDICES["Red"] + 1).astype(np.float32)  # +1 car rasterio est 1-indexed
        nir = src.read(BAND_INDICES["NIR"] + 1).astype(np.float32)
        profile = src.profile.copy()
    
    # Calcul du NDVI avec gestion de la division par zéro
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir - red) / (nir + red)
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clamp entre -1 et 1
    ndvi = np.clip(ndvi, -1.0, 1.0)
    
    # Mettre à jour le profil
    profile.update(dtype=np.float32, count=1)
    
    return ndvi, profile


def get_thermal(ir_path: str) -> Tuple[np.ndarray, dict]:
    """
    Récupère l'orthomosaïque thermique.
    
    Args:
        ir_path: Chemin de l'orthomosaïque IR
        
    Returns:
        (temp_data, profile) : données thermiques et profil rasterio
    """
    with rasterio.open(ir_path) as src:
        data = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    
    profile.update(dtype=np.float32, count=1)
    
    return data, profile


def prepare_orthomosaics_for_fijiyama(
    year: str = "2021",
    hours: List[str] = None,
    source_dir: str = SOURCE_DIR,
    output_dir: str = OUTPUT_DIR,
    verbose: bool = True
) -> Dict:
    """
    Prépare les orthomosaïques de toutes les heures pour Fijiyama.
    
    Exporte pour chaque heure :
        - {hour}_mean_MS.tif : moyenne des 6 bandes MS
        - {hour}_NDVI.tif : indice de végétation
        - {hour}_Temp.tif : thermique
    
    Args:
        year: Année à traiter
        hours: Liste des heures (par défaut toutes)
        source_dir: Répertoire source des orthomosaïques
        output_dir: Répertoire de sortie
        verbose: Afficher la progression
        
    Returns:
        Rapport d'exécution
    """
    if hours is None:
        hours = HOURS_2021 if year == "2021" else HOURS_2021
    
    # Créer le dossier de sortie
    out_dir = os.path.join(output_dir, f"fijiyama_{year}")
    os.makedirs(out_dir, exist_ok=True)
    
    report = {"year": year, "output_dir": out_dir, "hours": {}}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"PRÉPARATION DES ORTHOMOSAÏQUES {year} POUR FIJIYAMA")
        print(f"{'='*60}")
        print(f"Source: {source_dir}")
        print(f"Output: {out_dir}")
        print()
    
    for hour in hours:
        if verbose:
            print(f"📷 {hour}:", end=" ")
        
        hour_report = {"MS": False, "NDVI": False, "Temp": False}
        
        # Récupérer les chemins
        paths = get_orthomosaic_paths(year, hour, source_dir)
        
        if paths["MS"] is None:
            if verbose:
                print(f"⚠️ MS non trouvé")
            report["hours"][hour] = hour_report
            continue
        
        try:
            # 1. Mean MS
            mean_data, profile = compute_mean_ms(paths["MS"])
            mean_path = os.path.join(out_dir, f"{hour}_mean_MS.tif")
            with rasterio.open(mean_path, 'w', **profile) as dst:
                dst.write(mean_data, 1)
            hour_report["MS"] = True
            if verbose:
                print("MS✓", end=" ")
            
            # 2. NDVI
            ndvi_data, profile = compute_ndvi_from_bands(paths["MS"])
            ndvi_path = os.path.join(out_dir, f"{hour}_NDVI.tif")
            with rasterio.open(ndvi_path, 'w', **profile) as dst:
                dst.write(ndvi_data, 1)
            hour_report["NDVI"] = True
            if verbose:
                print("NDVI✓", end=" ")
            
            # 3. Thermique
            if paths["IR"]:
                temp_data, profile = get_thermal(paths["IR"])
                temp_path = os.path.join(out_dir, f"{hour}_Temp.tif")
                with rasterio.open(temp_path, 'w', **profile) as dst:
                    dst.write(temp_data, 1)
                hour_report["Temp"] = True
                if verbose:
                    print("Temp✓", end=" ")
            else:
                if verbose:
                    print("(no IR)", end=" ")
            
            if verbose:
                print()
                
        except Exception as e:
            if verbose:
                print(f"❌ Erreur: {e}")
        
        report["hours"][hour] = hour_report
    
    # Créer le README
    readme_path = os.path.join(out_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write(f"Orthomosaïques {year} préparées pour Fijiyama\n")
        f.write("=" * 50 + "\n\n")
        f.write("Fichiers par heure:\n")
        f.write("  - XX_mean_MS.tif : moyenne des 6 bandes MS\n")
        f.write("  - XX_NDVI.tif : indice de végétation\n")
        f.write("  - XX_Temp.tif : thermique\n\n")
        f.write(f"Heure de référence recommandée: {REFERENCE_HOUR}\n\n")
        f.write("Workflow Fijiyama:\n")
        f.write("1. Ouvrir les images mean_MS dans Fiji\n")
        f.write("2. Image > Adjust > Brightness/Contrast > Auto\n")
        f.write("3. Image > Type > 8-bit (pour réduire la taille)\n")
        f.write("4. Plugins > Fijiyama\n")
        f.write(f"5. Utiliser {REFERENCE_HOUR}_mean_MS.tif comme référence\n")
        f.write("6. Calculer les transformations\n")
        f.write("7. Exporter les matrices de transformation\n")
    
    if verbose:
        print(f"\n📝 README créé: {readme_path}")
    
    return report


def create_fiji_macro(output_dir: str, year: str = "2021") -> str:
    """
    Crée une macro Fiji pour la conversion 8-bit et préparation.
    
    Args:
        output_dir: Répertoire de sortie
        year: Année
        
    Returns:
        Chemin de la macro créée
    """
    macro_content = f'''// Macro Fiji pour préparer les orthomosaïques pour Fijiyama
// Projet DADY - {year}

dir = "{output_dir}/fijiyama_{year}/";
hours = newArray("09h", "11h", "13h", "15h", "17h");
types = newArray("mean_MS", "NDVI");

// Créer le dossier 8-bit
dir_8bit = dir + "8bit/";
File.makeDirectory(dir_8bit);

// Convertir chaque fichier
for (h = 0; h < hours.length; h++) {{
    for (t = 0; t < types.length; t++) {{
        filename = hours[h] + "_" + types[t] + ".tif";
        filepath = dir + filename;
        
        if (File.exists(filepath)) {{
            print("Processing: " + filename);
            open(filepath);
            
            // Auto-contrast
            run("Enhance Contrast", "saturated=0.35");
            
            // Convertir en 8-bit
            run("8-bit");
            
            // Sauvegarder
            saveAs("Tiff", dir_8bit + filename);
            close();
        }}
    }}
}}

print("Conversion terminée !");
print("Les fichiers 8-bit sont dans: " + dir_8bit);
print("Vous pouvez maintenant lancer Fijiyama avec ces fichiers.");
'''
    
    macro_path = os.path.join(output_dir, f"convert_to_8bit_{year}.ijm")
    with open(macro_path, 'w') as f:
        f.write(macro_content)
    
    return macro_path


def prepare_all_plots_for_fijiyama(*args, **kwargs):
    """Alias pour compatibilité - redirige vers prepare_orthomosaics_for_fijiyama."""
    return prepare_orthomosaics_for_fijiyama(*args, **kwargs)


def prepare_plot_for_fijiyama(*args, **kwargs):
    """Déprécié - utiliser prepare_orthomosaics_for_fijiyama."""
    print("⚠️ ATTENTION: prepare_plot_for_fijiyama est déprécié.")
    print("   Le recalage doit être fait sur les orthomosaïques complètes.")
    print("   Utilisez prepare_orthomosaics_for_fijiyama() à la place.")
    return None


def create_test_dataset():
    """
    Crée un dataset de test avec les orthomosaïques 2021.
    """
    print("=" * 60)
    print("CRÉATION DU DATASET FIJIYAMA")
    print("=" * 60)
    
    # Préparer les orthomosaïques
    report = prepare_orthomosaics_for_fijiyama(
        year="2021",
        hours=HOURS_2021,
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        verbose=True
    )
    
    # Créer la macro Fiji
    macro_path = create_fiji_macro(OUTPUT_DIR, "2021")
    print(f"\n📜 Macro Fiji créée: {macro_path}")
    
    # Afficher les infos sur les fichiers
    out_dir = os.path.join(OUTPUT_DIR, "fijiyama_2021")
    print(f"\n{'='*60}")
    print("FICHIERS CRÉÉS")
    print("=" * 60)
    
    for f in sorted(os.listdir(out_dir)):
        filepath = os.path.join(out_dir, f)
        if os.path.isfile(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.1f} MB")
    
    print(f"\n{'='*60}")
    print("PROCHAINES ÉTAPES")
    print("=" * 60)
    print("1. Ouvrir Fiji")
    print(f"2. Plugins > Macros > Run... > {macro_path}")
    print("   (convertit les fichiers en 8-bit)")
    print("3. Ouvrir Fijiyama")
    print(f"4. Charger les 5 images mean_MS depuis fijiyama_2021/8bit/")
    print(f"5. Définir {REFERENCE_HOUR}_mean_MS.tif comme référence")
    print("6. Calculer les transformations (Rigid ou Affine)")
    print("7. Exporter les matrices de transformation")
    
    return report


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    create_test_dataset()
