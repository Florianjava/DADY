"""
extract_2021.py - Extraction des données journalières pour l'année 2021

Ce module extrait pour chaque polygone du plan d'expérience :
- Les patches multispectraux (6 bandes)
- Les patches thermiques (1 bande)
- Les valeurs moyennes sur différentes zones (plante, sol, polygone entier)
- Le NDVI et la segmentation plante/sol
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from shapely.geometry import mapping
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_ROOT = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/anatomics"

# Horaires disponibles pour 2021
HOURS_2021 = ["09h", "11h", "13h", "15h", "17h"]

# Noms des bandes MS (6 bandes)
MS_BAND_NAMES = ["Blue", "Green", "Red", "RedEdge", "NIR", "Band6"]

# Seuil NDVI pour segmentation plante/sol
NDVI_THRESHOLD = 0.0


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def excel_to_shp_plot(plot_str: str) -> int:
    """
    Convertit un identifiant Plot Excel (Pa1, Pa42, Pa640...) 
    en identifiant Plot Shapefile (1001, 1042, 4160...).
    
    Correspondance:
        R1 (Pa1-Pa160)   -> 1001-1160
        R2 (Pa161-Pa320) -> 2001-2160
        R3 (Pa321-Pa480) -> 3001-3160
        R4 (Pa481-Pa640) -> 4001-4160
    """
    num = int(plot_str.replace('Pa', ''))
    if num <= 160:
        return 1000 + num
    elif num <= 320:
        return 2000 + (num - 160)
    elif num <= 480:
        return 3000 + (num - 320)
    else:
        return 4000 + (num - 480)


def shp_to_excel_plot(shp_plot: int) -> str:
    """
    Convertit un identifiant Plot Shapefile (1001, 1042, 4160...)
    en identifiant Plot Excel (Pa1, Pa42, Pa640...).
    """
    thousands = shp_plot // 1000
    remainder = shp_plot % 1000
    
    if thousands == 1:
        return f"Pa{remainder}"
    elif thousands == 2:
        return f"Pa{160 + remainder}"
    elif thousands == 3:
        return f"Pa{320 + remainder}"
    else:
        return f"Pa{480 + remainder}"


def get_file_paths(hour: str, year: int = 2021) -> Dict[str, str]:
    """
    Retourne les chemins des fichiers pour une heure donnée.
    
    Attention: pour 09h, le nom du fichier IR est "2021_09h_IR.tif"
    mais le shapefile est "2021_9h_STR.shp" (sans le zéro)
    """
    # Normalisation de l'heure pour les noms de fichiers
    hour_normalized = hour.replace("0", "") if hour.startswith("0") else hour
    
    base = DATA_ROOT
    
    paths = {
        "ms": os.path.join(base, f"{year}_{hour}_MS.tif"),
        "ir": os.path.join(base, f"{year}_{hour}_IR.tif"),
        "shp_str": os.path.join(base, f"{year}_{hour_normalized}_STR.shp"),
        "shp_etm": os.path.join(base, f"{year}_{hour_normalized}_ETM.shp"),
        "randomisation": os.path.join(base, f"Anatomics_Randomisation_{year}.xlsx"),
    }
    
    # Vérifier l'existence et corriger si nécessaire
    if not os.path.exists(paths["ms"]):
        # Essayer sans le zéro initial
        paths["ms"] = os.path.join(base, f"{year}_{hour_normalized}_MS.tif")
    if not os.path.exists(paths["ir"]):
        paths["ir"] = os.path.join(base, f"{year}_{hour_normalized}_IR.tif")
    
    return paths


def compute_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calcule le NDVI pixel par pixel.
    
    NDVI = (NIR - Red) / (NIR + Red)
    
    Args:
        red: Array des valeurs de la bande rouge
        nir: Array des valeurs de la bande NIR
        
    Returns:
        Array NDVI (valeurs entre -1 et 1)
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (nir.astype(float) - red.astype(float)) / (nir.astype(float) + red.astype(float))
        ndvi = np.nan_to_num(ndvi, nan=0.0, posinf=0.0, neginf=0.0)
    return ndvi


def export_patch_info_csv(output_dir: str, polygon_id: str, info: Dict, stats: Dict):
    """
    Exporte un fichier CSV avec les informations du patch.
    """
    csv_path = os.path.join(output_dir, f"{polygon_id}_info.csv")
    
    # Combiner info et stats
    data = {**info, **stats}
    
    # Supprimer bbox qui est un tuple (pas pratique pour CSV)
    if 'bbox' in data:
        bbox = data.pop('bbox')
        data['bbox_minx'] = bbox[0]
        data['bbox_miny'] = bbox[1]
        data['bbox_maxx'] = bbox[2]
        data['bbox_maxy'] = bbox[3]
    
    # Écrire le CSV
    df = pd.DataFrame([data])
    df.to_csv(csv_path, index=False)


def extract_patch_and_stats(
    ms_src: rasterio.DatasetReader,
    ir_src: rasterio.DatasetReader,
    geometry,
    output_dir: Optional[str] = None,
    polygon_id: str = "",
    polygon_info: Optional[Dict] = None
) -> Dict:
    """
    Extrait le patch et calcule les statistiques pour un polygone donné.
    
    Args:
        ms_src: Dataset rasterio ouvert pour l'image MS
        ir_src: Dataset rasterio ouvert pour l'image IR (peut être None)
        geometry: Géométrie du polygone (shapely)
        output_dir: Répertoire de sortie pour les patches (optionnel)
        polygon_id: Identifiant du polygone pour le nommage
        polygon_info: Informations du polygone (geno, env, etc.) pour le CSV
        
    Returns:
        Dictionnaire contenant toutes les statistiques calculées
    """
    results = {}
    
    # Reprojection de la géométrie si nécessaire
    geom = geometry
    
    # Bounding box du polygone
    minx, miny, maxx, maxy = geom.bounds
    results["bbox"] = (minx, miny, maxx, maxy)
    
    # Conversion en coordonnées pixels pour MS
    row_min, col_min = ms_src.index(minx, maxy)
    row_max, col_max = ms_src.index(maxx, miny)
    
    # Assurer l'ordre correct
    row_min, row_max = min(row_min, row_max), max(row_min, row_max)
    col_min, col_max = min(col_min, col_max), max(col_min, col_max)
    
    height = row_max - row_min
    width = col_max - col_min
    
    if height <= 0 or width <= 0:
        return None
    
    # Fenêtre d'extraction MS
    window_ms = Window(col_min, row_min, width, height)
    win_transform_ms = ms_src.window_transform(window_ms)
    
    # Lecture des bandes MS (6 bandes)
    ms_data = ms_src.read(window=window_ms, boundless=True, fill_value=0)
    
    # Création du masque du polygone (S_poly)
    mask_poly = rasterize(
        [(mapping(geom), 1)],
        out_shape=(height, width),
        transform=win_transform_ms,
        fill=0,
        all_touched=True,
        dtype=np.uint8
    )
    
    # Calcul du NDVI (Red = bande 3, NIR = bande 5)
    red_band = ms_data[2]  # Index 2 = bande 3 (Red)
    nir_band = ms_data[4]  # Index 4 = bande 5 (NIR)
    ndvi = compute_ndvi(red_band, nir_band)
    
    # Segmentation plante/sol basée sur NDVI
    mask_plant = ((ndvi > NDVI_THRESHOLD) & (mask_poly > 0)).astype(np.uint8)
    mask_soil = ((ndvi <= NDVI_THRESHOLD) & (mask_poly > 0)).astype(np.uint8)
    
    # Surfaces (en pixels)
    results["surface_poly"] = int(np.sum(mask_poly > 0))
    results["surface_plant"] = int(np.sum(mask_plant > 0))
    results["surface_soil"] = int(np.sum(mask_soil > 0))
    
    # Moyennes MS sur chaque zone (6 bandes)
    for i, band_name in enumerate(MS_BAND_NAMES):
        band_data = ms_data[i].astype(float)
        
        # Moyenne sur S_poly
        if results["surface_poly"] > 0:
            results[f"mean_{band_name}_poly"] = float(np.mean(band_data[mask_poly > 0]))
        else:
            results[f"mean_{band_name}_poly"] = np.nan
        
        # Moyenne sur S_plant
        if results["surface_plant"] > 0:
            results[f"mean_{band_name}_plant"] = float(np.mean(band_data[mask_plant > 0]))
        else:
            results[f"mean_{band_name}_plant"] = np.nan
        
        # Moyenne sur S_soil
        if results["surface_soil"] > 0:
            results[f"mean_{band_name}_soil"] = float(np.mean(band_data[mask_soil > 0]))
        else:
            results[f"mean_{band_name}_soil"] = np.nan
    
    # Moyenne NDVI
    if results["surface_poly"] > 0:
        results["mean_NDVI_poly"] = float(np.mean(ndvi[mask_poly > 0]))
    else:
        results["mean_NDVI_poly"] = np.nan
    if results["surface_plant"] > 0:
        results["mean_NDVI_plant"] = float(np.mean(ndvi[mask_plant > 0]))
    else:
        results["mean_NDVI_plant"] = np.nan
    if results["surface_soil"] > 0:
        results["mean_NDVI_soil"] = float(np.mean(ndvi[mask_soil > 0]))
    else:
        results["mean_NDVI_soil"] = np.nan
    
    # Traitement IR (thermique) si disponible
    if ir_src is not None:
        # L'IR peut avoir une résolution différente, on calcule sa propre fenêtre
        try:
            ir_row_min, ir_col_min = ir_src.index(minx, maxy)
            ir_row_max, ir_col_max = ir_src.index(maxx, miny)
            ir_row_min, ir_row_max = min(ir_row_min, ir_row_max), max(ir_row_min, ir_row_max)
            ir_col_min, ir_col_max = min(ir_col_min, ir_col_max), max(ir_col_min, ir_col_max)
            
            ir_height = ir_row_max - ir_row_min
            ir_width = ir_col_max - ir_col_min
            
            if ir_height > 0 and ir_width > 0:
                window_ir = Window(ir_col_min, ir_row_min, ir_width, ir_height)
                win_transform_ir = ir_src.window_transform(window_ir)
                
                ir_data = ir_src.read(1, window=window_ir, boundless=True, fill_value=0)
                
                # Masque IR (résolution différente)
                mask_poly_ir = rasterize(
                    [(mapping(geom), 1)],
                    out_shape=(ir_height, ir_width),
                    transform=win_transform_ir,
                    fill=0,
                    all_touched=True,
                    dtype=np.uint8
                )
                
                # Pour les masques plant/soil en IR, on recalcule avec la résolution IR
                # ou on utilise une approximation basée sur le ratio de surface
                surface_ir_poly = np.sum(mask_poly_ir > 0)
                
                if surface_ir_poly > 0:
                    results["mean_Temp_poly"] = float(np.mean(ir_data[mask_poly_ir > 0]))
                else:
                    results["mean_Temp_poly"] = np.nan
                
                # Pour plant/soil en IR, on applique un ratio approximatif
                # basé sur les surfaces MS (simplification)
                plant_ratio = results["surface_plant"] / max(results["surface_poly"], 1)
                soil_ratio = results["surface_soil"] / max(results["surface_poly"], 1)
                
                # Utilisation du NDVI recalculé sur la grille IR pour une meilleure précision
                # serait plus précis mais demande les bandes MS rééchantillonnées
                # Pour simplifier, on garde la moyenne globale pour l'instant
                results["mean_Temp_plant"] = results["mean_Temp_poly"]  # Approximation
                results["mean_Temp_soil"] = results["mean_Temp_poly"]   # Approximation
            else:
                results["mean_Temp_poly"] = np.nan
                results["mean_Temp_plant"] = np.nan
                results["mean_Temp_soil"] = np.nan
        except Exception as e:
            results["mean_Temp_poly"] = np.nan
            results["mean_Temp_plant"] = np.nan
            results["mean_Temp_soil"] = np.nan
    else:
        results["mean_Temp_poly"] = np.nan
        results["mean_Temp_plant"] = np.nan
        results["mean_Temp_soil"] = np.nan
    
    # Export des patches si demandé
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        
        # Export canal par canal pour MS
        for i, band_name in enumerate(MS_BAND_NAMES):
            band_path = os.path.join(output_dir, f"{polygon_id}_{band_name}.tif")
            with rasterio.open(
                band_path, 'w', driver='GTiff',
                height=height, width=width,
                count=1,
                dtype=ms_data.dtype,
                crs=ms_src.crs,
                transform=win_transform_ms
            ) as dst:
                dst.write(ms_data[i], 1)
        
        # Export NDVI
        ndvi_path = os.path.join(output_dir, f"{polygon_id}_NDVI.tif")
        with rasterio.open(
            ndvi_path, 'w', driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=np.float32,
            crs=ms_src.crs,
            transform=win_transform_ms
        ) as dst:
            dst.write(ndvi.astype(np.float32), 1)
        
        # Patch IR si disponible
        if ir_src is not None:
            try:
                ir_row_min_export, ir_col_min_export = ir_src.index(minx, maxy)
                ir_row_max_export, ir_col_max_export = ir_src.index(maxx, miny)
                ir_row_min_export, ir_row_max_export = min(ir_row_min_export, ir_row_max_export), max(ir_row_min_export, ir_row_max_export)
                ir_col_min_export, ir_col_max_export = min(ir_col_min_export, ir_col_max_export), max(ir_col_min_export, ir_col_max_export)
                
                ir_height_export = ir_row_max_export - ir_row_min_export
                ir_width_export = ir_col_max_export - ir_col_min_export
                
                if ir_height_export > 0 and ir_width_export > 0:
                    window_ir_export = Window(ir_col_min_export, ir_row_min_export, ir_width_export, ir_height_export)
                    win_transform_ir_export = ir_src.window_transform(window_ir_export)
                    ir_data_export = ir_src.read(1, window=window_ir_export, boundless=True, fill_value=0)
                    
                    ir_path = os.path.join(output_dir, f"{polygon_id}_Temp.tif")
                    with rasterio.open(
                        ir_path, 'w', driver='GTiff',
                        height=ir_height_export, width=ir_width_export,
                        count=1,
                        dtype=ir_data_export.dtype,
                        crs=ir_src.crs,
                        transform=win_transform_ir_export
                    ) as dst:
                        dst.write(ir_data_export, 1)
            except Exception as e:
                pass
        
        # Export des masques
        mask_poly_path = os.path.join(output_dir, f"{polygon_id}_mask_poly.tif")
        with rasterio.open(
            mask_poly_path, 'w', driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=np.uint8,
            crs=ms_src.crs,
            transform=win_transform_ms
        ) as dst:
            dst.write(mask_poly, 1)
        
        mask_plant_path = os.path.join(output_dir, f"{polygon_id}_mask_plant.tif")
        with rasterio.open(
            mask_plant_path, 'w', driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=np.uint8,
            crs=ms_src.crs,
            transform=win_transform_ms
        ) as dst:
            dst.write(mask_plant, 1)
        
        mask_soil_path = os.path.join(output_dir, f"{polygon_id}_mask_soil.tif")
        with rasterio.open(
            mask_soil_path, 'w', driver='GTiff',
            height=height, width=width,
            count=1,
            dtype=np.uint8,
            crs=ms_src.crs,
            transform=win_transform_ms
        ) as dst:
            dst.write(mask_soil, 1)
        
        # Export du CSV d'information du patch
        if polygon_info is not None:
            export_patch_info_csv(output_dir, polygon_id, polygon_info, results)
    
    return results


def extract_daily_data_2021(
    hours: List[str] = None,
    output_csv: str = "daily_sahel_2021.csv",
    output_patches_dir: Optional[str] = None,
    shapefile_type: str = "STR",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extrait les données journalières pour toutes les heures et tous les polygones.
    
    Args:
        hours: Liste des heures à traiter (par défaut: toutes)
        output_csv: Chemin du fichier CSV de sortie
        output_patches_dir: Répertoire pour les patches (None = pas d'export)
        shapefile_type: "STR" ou "ETM" selon le type de shapefile
        verbose: Afficher la progression
        
    Returns:
        DataFrame avec toutes les données extraites
    """
    if hours is None:
        hours = HOURS_2021
    
    # Charger le fichier de randomisation
    randomisation_path = os.path.join(DATA_ROOT, "Anatomics_Randomisation_2021.xlsx")
    rand_df = pd.read_excel(randomisation_path)
    
    # Créer un dictionnaire pour lookup rapide
    rand_dict = {}
    for _, row in rand_df.iterrows():
        shp_plot = excel_to_shp_plot(row['Plot'])
        rand_dict[shp_plot] = {
            'plot_excel': row['Plot'],
            'env': row['Regime'],
            'repetition': row['Repetition'],
            'bloc': row['Bloc'],
            'geno': row['Lines']
        }
    
    all_results = []
    
    for hour in hours:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Traitement de l'heure: {hour}")
            print(f"{'='*60}")
        
        paths = get_file_paths(hour)
        
        # Vérifier l'existence des fichiers
        if not os.path.exists(paths["ms"]):
            if verbose:
                print(f"⚠️ Fichier MS non trouvé: {paths['ms']}")
            continue
        
        # Charger le shapefile
        shp_key = f"shp_{shapefile_type.lower()}"
        if not os.path.exists(paths[shp_key]):
            if verbose:
                print(f"⚠️ Shapefile non trouvé: {paths[shp_key]}")
            continue
        
        shp = gpd.read_file(paths[shp_key])
        
        if verbose:
            print(f"📂 MS: {paths['ms']}")
            print(f"📂 IR: {paths['ir']}")
            print(f"📂 SHP: {paths[shp_key]}")
            print(f"📊 {len(shp)} polygones à traiter")
        
        # Ouvrir les rasters
        ms_src = rasterio.open(paths["ms"])
        ir_src = None
        if os.path.exists(paths["ir"]):
            ir_src = rasterio.open(paths["ir"])
        
        # Répertoire de sortie pour cette heure
        hour_output_dir = None
        if output_patches_dir:
            hour_output_dir = os.path.join(output_patches_dir, f"2021_{hour}")
        
        # Traiter chaque polygone
        for idx, row in shp.iterrows():
            shp_plot = row['Plot']
            
            # Récupérer les infos du CSV
            if shp_plot not in rand_dict:
                if verbose and idx < 5:
                    print(f"⚠️ Plot {shp_plot} non trouvé dans randomisation")
                continue
            
            info = rand_dict[shp_plot]
            polygon_id = f"{info['plot_excel']}_{hour}"
            
            # Extraire les statistiques
            geom = row.geometry
            
            # Reprojection si nécessaire (shapefile en UTM, raster en WGS84)
            if shp.crs != ms_src.crs:
                geom = gpd.GeoSeries([geom], crs=shp.crs).to_crs(ms_src.crs).iloc[0]
            
            # Préparer les infos pour le CSV du patch
            patch_info = {
                'hour': hour,
                'plot_shp': shp_plot,
                'plot_excel': info['plot_excel'],
                'env': info['env'],
                'repetition': info['repetition'],
                'bloc': info['bloc'],
                'geno': info['geno'],
            }
            
            stats = extract_patch_and_stats(
                ms_src, ir_src, geom,
                output_dir=hour_output_dir,
                polygon_id=polygon_id,
                polygon_info=patch_info
            )
            
            if stats is None:
                continue
            
            # Construire la ligne de résultat
            result = {
                'hour': hour,
                'plot_shp': shp_plot,
                'plot_excel': info['plot_excel'],
                'env': info['env'],
                'repetition': info['repetition'],
                'bloc': info['bloc'],
                'geno': info['geno'],
            }
            result.update(stats)
            
            all_results.append(result)
            
            if verbose and (idx + 1) % 100 == 0:
                print(f"  ✓ {idx + 1}/{len(shp)} polygones traités")
        
        # Fermer les fichiers
        ms_src.close()
        if ir_src:
            ir_src.close()
        
        if verbose:
            print(f"✅ {hour} terminé: {len([r for r in all_results if r['hour'] == hour])} polygones")
    
    # Créer le DataFrame final
    df = pd.DataFrame(all_results)
    
    # Sauvegarder en CSV
    if output_csv:
        df.to_csv(output_csv, index=False)
        if verbose:
            print(f"\n💾 Résultats sauvegardés dans: {output_csv}")
            print(f"📊 Total: {len(df)} lignes ({len(df['hour'].unique())} heures × {len(df['plot_excel'].unique())} polygones)")
    
    return df


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Exemple d'utilisation
    df = extract_daily_data_2021(
        hours=["11h"],  # Pour tester, commencer avec une seule heure
        output_csv="daily_sahel_2021_test.csv",
        output_patches_dir=None,  # Pas d'export de patches pour le test
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Aperçu des résultats:")
    print("="*60)
    print(df.head(10))
    print(f"\nColonnes: {list(df.columns)}")
