#!/usr/bin/env python3
"""
align_pairwise.py - Recalage pairwise avec points de contrôle

Calcule et applique une transformation affine pour aligner chaque heure
sur l'image de référence (13h) à partir de points de correspondance.

Format CSV des points (un fichier par paire):
    x_13h,y_13h,x_other,y_other
    1200,1300,1250,1340
    3400,1250,3420,1280
    ...

Fichiers attendus:
    - points_09h.csv (correspondances 13h <-> 09h)
    - points_11h.csv (correspondances 13h <-> 11h)
    - points_15h.csv (correspondances 13h <-> 15h)
    - points_17h.csv (correspondances 13h <-> 17h)

Usage:
    python -m registration.align_pairwise
"""

import os
import numpy as np
import pandas as pd
import rasterio
from scipy import ndimage
from typing import Tuple, List, Dict, Optional
from pathlib import Path


# =============================================================================
# CONFIGURATION
# =============================================================================

# Répertoire de travail
WORK_DIR = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"

# Dossier des crops
CROPS_DIR = os.path.join(WORK_DIR, "crops_2021")

# Dossier de sortie pour les images alignées
OUTPUT_DIR = os.path.join(WORK_DIR, "aligned_2021")

# Heures à recaler (13h est la référence)
HOURS_TO_ALIGN = ["09h", "11h", "15h", "17h"]
REFERENCE_HOUR = "13h"


# =============================================================================
# CALCUL DE LA TRANSFORMATION AFFINE
# =============================================================================

def compute_affine_transform(
    pts_ref: np.ndarray,
    pts_src: np.ndarray
) -> np.ndarray:
    """
    Calcule la matrice de transformation affine 2D.
    
    Trouve la transformation T telle que: pts_ref = T @ pts_src
    
    Args:
        pts_ref: Points de référence (N, 2) - coordonnées (x, y) sur 13h
        pts_src: Points source (N, 2) - coordonnées (x, y) sur l'autre image
        
    Returns:
        Matrice affine 3x3 (coordonnées homogènes)
    """
    n_points = pts_ref.shape[0]
    
    if n_points < 3:
        raise ValueError("Il faut au moins 3 points pour calculer une transformation affine")
    
    # Construction du système linéaire pour transformation affine
    # [x'] = [a b tx] [x]
    # [y']   [c d ty] [y]
    # [1 ]   [0 0 1 ] [1]
    
    # On résout: pts_ref = M @ pts_src (en coordonnées homogènes)
    # Avec M = [[a, b, tx], [c, d, ty], [0, 0, 1]]
    
    # Système surdéterminé résolu par moindres carrés
    A = np.zeros((2 * n_points, 6))
    b = np.zeros(2 * n_points)
    
    for i in range(n_points):
        x, y = pts_src[i]
        x_ref, y_ref = pts_ref[i]
        
        # Équation pour x'
        A[2*i, 0] = x
        A[2*i, 1] = y
        A[2*i, 2] = 1
        b[2*i] = x_ref
        
        # Équation pour y'
        A[2*i+1, 3] = x
        A[2*i+1, 4] = y
        A[2*i+1, 5] = 1
        b[2*i+1] = y_ref
    
    # Résolution par moindres carrés
    params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Construction de la matrice 3x3
    M = np.array([
        [params[0], params[1], params[2]],
        [params[3], params[4], params[5]],
        [0, 0, 1]
    ])
    
    return M


def compute_transform_error(
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    M: np.ndarray
) -> Dict:
    """
    Calcule l'erreur de transformation (résidus).
    
    Returns:
        Dict avec RMSE, erreur max, erreurs par point
    """
    n_points = pts_ref.shape[0]
    errors = []
    
    for i in range(n_points):
        # Transformer le point source
        pt_src_h = np.array([pts_src[i, 0], pts_src[i, 1], 1])
        pt_transformed = M @ pt_src_h
        
        # Erreur par rapport au point de référence
        error = np.sqrt((pt_transformed[0] - pts_ref[i, 0])**2 + 
                       (pt_transformed[1] - pts_ref[i, 1])**2)
        errors.append(error)
    
    errors = np.array(errors)
    
    return {
        "rmse": np.sqrt(np.mean(errors**2)),
        "max_error": np.max(errors),
        "mean_error": np.mean(errors),
        "errors_per_point": errors
    }


# =============================================================================
# APPLICATION DE LA TRANSFORMATION
# =============================================================================

def apply_affine_to_image(
    src_path: str,
    dst_path: str,
    M: np.ndarray,
    output_shape: Tuple[int, int] = None,
    order: int = 1,
    cval: float = 0
) -> bool:
    """
    Applique une transformation affine à une image.
    
    Args:
        src_path: Image source
        dst_path: Image destination
        M: Matrice affine 3x3 (src -> ref)
        output_shape: Taille de sortie (height, width). Si None, même taille que source
        order: Ordre d'interpolation (0=nearest, 1=bilinear, 3=cubic)
        cval: Valeur pour les pixels hors image
        
    Returns:
        True si succès
    """
    try:
        with rasterio.open(src_path) as src:
            data = src.read()
            profile = src.profile.copy()
            src_shape = (src.height, src.width)
        
        if output_shape is None:
            output_shape = src_shape
        
        # Pour ndimage.affine_transform, on a besoin de la transformation inverse
        # car elle mappe les coordonnées de sortie vers les coordonnées d'entrée
        M_inv = np.linalg.inv(M)
        
        # Extraire la partie 2x2 et le vecteur de translation
        # ndimage utilise le format: output[i,j] = input[M[0,0]*i + M[0,1]*j + offset[0], ...]
        # Attention: ndimage travaille en (row, col) = (y, x)
        
        # Conversion pour ndimage (y, x) au lieu de (x, y)
        # M transforme (x, y) -> on doit transformer (y, x)
        M_yx = np.array([
            [M_inv[1, 1], M_inv[1, 0], M_inv[1, 2]],
            [M_inv[0, 1], M_inv[0, 0], M_inv[0, 2]],
            [0, 0, 1]
        ])
        
        matrix = M_yx[:2, :2]
        offset = M_yx[:2, 2]
        
        # Appliquer à chaque bande
        n_bands = data.shape[0]
        output_data = np.zeros((n_bands, output_shape[0], output_shape[1]), dtype=data.dtype)
        
        for b in range(n_bands):
            output_data[b] = ndimage.affine_transform(
                data[b].astype(np.float64),
                matrix,
                offset=offset,
                output_shape=output_shape,
                order=order,
                mode='constant',
                cval=cval
            ).astype(data.dtype)
        
        # Mettre à jour le profil
        profile.update(
            height=output_shape[0],
            width=output_shape[1]
        )
        
        # Sauvegarder
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        with rasterio.open(dst_path, 'w', **profile) as dst:
            dst.write(output_data)
        
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# CHARGEMENT DES POINTS
# =============================================================================

def load_control_points(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les points de contrôle depuis un CSV.
    
    Format attendu:
        x_13h,y_13h,x_other,y_other
        1200,1300,1250,1340
        ...
    
    Returns:
        (pts_ref, pts_src) : arrays (N, 2)
    """
    df = pd.read_csv(csv_path)
    
    # Détecter les colonnes
    if 'x_13h' in df.columns:
        pts_ref = df[['x_13h', 'y_13h']].values
        pts_src = df[['x_other', 'y_other']].values
    elif 'x_ref' in df.columns:
        pts_ref = df[['x_ref', 'y_ref']].values
        pts_src = df[['x_src', 'y_src']].values
    else:
        # Supposer que les colonnes sont dans l'ordre
        pts_ref = df.iloc[:, 0:2].values
        pts_src = df.iloc[:, 2:4].values
    
    return pts_ref, pts_src


def load_control_points_from_two_files(
    ref_csv: str, 
    src_csv: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Charge les points de contrôle depuis deux fichiers CSV séparés.
    
    Format attendu (même nombre de lignes, même ordre):
        x,y
        1200,1300
        ...
    
    Args:
        ref_csv: CSV avec les points de référence (13h)
        src_csv: CSV avec les points source (autre heure)
        
    Returns:
        (pts_ref, pts_src) : arrays (N, 2)
    """
    df_ref = pd.read_csv(ref_csv)
    df_src = pd.read_csv(src_csv)
    
    pts_ref = df_ref[['x', 'y']].values
    pts_src = df_src[['x', 'y']].values
    
    if len(pts_ref) != len(pts_src):
        raise ValueError(f"Nombre de points différent: {len(pts_ref)} vs {len(pts_src)}")
    
    return pts_ref, pts_src


# =============================================================================
# WORKFLOW PRINCIPAL
# =============================================================================

def align_hour_to_reference(
    hour: str,
    points_csv: str,
    crops_dir: str = CROPS_DIR,
    output_dir: str = OUTPUT_DIR,
    reference_hour: str = REFERENCE_HOUR,
    image_types: List[str] = ["mean_MS", "NDVI", "MS_crop", "Temp_crop"],
    verbose: bool = True
) -> Dict:
    """
    Aligne toutes les images d'une heure sur la référence.
    
    Args:
        hour: Heure à aligner (ex: "09h")
        points_csv: Chemin du CSV avec les points de correspondance
        crops_dir: Dossier des crops
        output_dir: Dossier de sortie
        reference_hour: Heure de référence
        image_types: Types d'images à traiter
        verbose: Afficher la progression
        
    Returns:
        Rapport d'exécution
    """
    report = {"hour": hour, "transform": None, "error": None, "images": []}
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"ALIGNEMENT {hour} -> {reference_hour}")
        print(f"{'='*60}")
    
    # 1. Charger les points
    if verbose:
        print(f"📍 Chargement des points: {points_csv}")
    
    pts_ref, pts_src = load_control_points(points_csv)
    n_points = pts_ref.shape[0]
    
    if verbose:
        print(f"   {n_points} points de correspondance")
    
    # 2. Calculer la transformation
    M = compute_affine_transform(pts_ref, pts_src)
    report["transform"] = M.tolist()
    
    if verbose:
        print(f"\n📐 Matrice de transformation:")
        print(f"   [{M[0,0]:.6f}, {M[0,1]:.6f}, {M[0,2]:.2f}]")
        print(f"   [{M[1,0]:.6f}, {M[1,1]:.6f}, {M[1,2]:.2f}]")
    
    # 3. Calculer l'erreur
    error_stats = compute_transform_error(pts_ref, pts_src, M)
    report["error"] = error_stats
    
    if verbose:
        print(f"\n📊 Erreur de recalage:")
        print(f"   RMSE: {error_stats['rmse']:.2f} pixels")
        print(f"   Max:  {error_stats['max_error']:.2f} pixels")
    
    # 4. Récupérer la taille de l'image de référence
    ref_path = os.path.join(crops_dir, f"{reference_hour}_mean_MS.tif")
    with rasterio.open(ref_path) as src:
        ref_shape = (src.height, src.width)
    
    if verbose:
        print(f"\n📷 Taille de référence: {ref_shape[1]}x{ref_shape[0]}")
    
    # 5. Appliquer aux images
    os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n🔄 Application de la transformation:")
    
    for img_type in image_types:
        src_path = os.path.join(crops_dir, f"{hour}_{img_type}.tif")
        
        if not os.path.exists(src_path):
            continue
        
        dst_path = os.path.join(output_dir, f"{hour}_{img_type}.tif")
        
        # Choisir l'interpolation selon le type
        order = 0 if "mask" in img_type.lower() else 1
        
        if verbose:
            print(f"   {img_type}:", end=" ")
        
        success = apply_affine_to_image(
            src_path, dst_path, M,
            output_shape=ref_shape,
            order=order
        )
        
        if success:
            if verbose:
                print("✅")
            report["images"].append(img_type)
        else:
            if verbose:
                print("❌")
    
    return report


def align_all_hours(
    points_dir: str = WORK_DIR,
    crops_dir: str = CROPS_DIR,
    output_dir: str = OUTPUT_DIR,
    hours: List[str] = HOURS_TO_ALIGN,
    verbose: bool = True
) -> Dict:
    """
    Aligne toutes les heures sur la référence.
    
    Cherche les fichiers points_XXh.csv dans points_dir.
    """
    reports = {}
    
    # Copier les images de référence
    ref_dir = output_dir
    os.makedirs(ref_dir, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print("COPIE DES IMAGES DE RÉFÉRENCE (13h)")
        print(f"{'='*60}")
    
    for img_type in ["mean_MS", "NDVI", "MS_crop", "Temp_crop"]:
        src = os.path.join(crops_dir, f"13h_{img_type}.tif")
        dst = os.path.join(ref_dir, f"13h_{img_type}.tif")
        if os.path.exists(src):
            import shutil
            shutil.copy2(src, dst)
            if verbose:
                print(f"   13h_{img_type} ✅")
    
    # Aligner chaque heure
    for hour in hours:
        # Chercher le fichier de points
        points_patterns = [
            os.path.join(points_dir, f"points_{hour}.csv"),
            os.path.join(points_dir, f"points_13h_{hour}.csv"),
            os.path.join(points_dir, f"{hour}_points.csv"),
        ]
        
        points_csv = None
        for pattern in points_patterns:
            if os.path.exists(pattern):
                points_csv = pattern
                break
        
        if points_csv is None:
            if verbose:
                print(f"\n⚠️ Pas de fichier de points trouvé pour {hour}")
                print(f"   Attendu: points_{hour}.csv")
            continue
        
        report = align_hour_to_reference(
            hour=hour,
            points_csv=points_csv,
            crops_dir=crops_dir,
            output_dir=output_dir,
            verbose=verbose
        )
        
        reports[hour] = report
    
    # Sauvegarder le rapport
    report_path = os.path.join(output_dir, "alignment_report.json")
    import json
    
    # Convertir numpy arrays pour JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(report_path, 'w') as f:
        json.dump(convert_for_json(reports), f, indent=2)
    
    if verbose:
        print(f"\n📝 Rapport sauvegardé: {report_path}")
    
    return reports


def create_points_template(hour: str, output_path: str = None) -> str:
    """
    Crée un template CSV pour les points de correspondance.
    """
    if output_path is None:
        output_path = os.path.join(WORK_DIR, f"points_{hour}.csv")
    
    template = """x_13h,y_13h,x_other,y_other
# Point 1 (exemple: coin haut-gauche d'un plot)
1200,1300,1250,1340
# Point 2 (exemple: coin haut-droit)
3400,1250,3420,1280
# Point 3 (exemple: coin bas-gauche)
1150,4490,1180,4520
# Point 4 (exemple: coin bas-droit)
3330,4440,3350,4460
# Ajouter d'autres points si nécessaire (minimum 3)
"""
    
    with open(output_path, 'w') as f:
        f.write(template)
    
    print(f"📝 Template créé: {output_path}")
    return output_path


# =============================================================================
# SCRIPT PRINCIPAL
# =============================================================================

def test_align_09h_ndvi_on_13h():
    """
    Test d'alignement de 09h NDVI sur 13h NDVI.
    Utilise les fichiers RoiSet_13_9ndvi_set1.csv (13h) et set2.csv (09h).
    """
    print("=" * 60)
    print("TEST: ALIGNEMENT 09h NDVI -> 13h NDVI")
    print("=" * 60)
    
    # Chemins des fichiers
    rois_dir = os.path.join(WORK_DIR, "Rois_0_align_crop_affine")
    ref_csv = os.path.join(rois_dir, "RoiSet_13_9ndvi_set1.csv")  # 13h
    src_csv = os.path.join(rois_dir, "RoiSet_13_9ndvi_set2.csv")  # 09h
    
    # Images
    src_image = os.path.join(CROPS_DIR, "09h_NDVI.tif")
    ref_image = os.path.join(CROPS_DIR, "13h_NDVI.tif")
    out_image = os.path.join(OUTPUT_DIR, "09h_NDVI_aligned.tif")
    
    # Vérifier que les fichiers existent
    for f in [ref_csv, src_csv, src_image, ref_image]:
        if not os.path.exists(f):
            print(f"❌ Fichier manquant: {f}")
            return
    
    # 1. Charger les points
    print(f"\n📍 Chargement des points...")
    pts_ref, pts_src = load_control_points_from_two_files(ref_csv, src_csv)
    print(f"   {len(pts_ref)} points de correspondance")
    
    # Afficher quelques points
    print(f"\n   Exemples de correspondances:")
    for i in range(min(3, len(pts_ref))):
        print(f"      13h: ({pts_ref[i,0]:.0f}, {pts_ref[i,1]:.0f}) <-> 09h: ({pts_src[i,0]:.0f}, {pts_src[i,1]:.0f})")
    
    # 2. Calculer la transformation
    print(f"\n📐 Calcul de la transformation affine...")
    M = compute_affine_transform(pts_ref, pts_src)
    
    print(f"\n   Matrice de transformation (09h -> 13h):")
    print(f"   | {M[0,0]:8.5f}  {M[0,1]:8.5f}  {M[0,2]:8.2f} |")
    print(f"   | {M[1,0]:8.5f}  {M[1,1]:8.5f}  {M[1,2]:8.2f} |")
    print(f"   | {M[2,0]:8.5f}  {M[2,1]:8.5f}  {M[2,2]:8.2f} |")
    
    # 3. Calculer l'erreur
    print(f"\n📊 Qualité du recalage:")
    error_stats = compute_transform_error(pts_ref, pts_src, M)
    print(f"   RMSE:       {error_stats['rmse']:.2f} pixels")
    print(f"   Erreur max: {error_stats['max_error']:.2f} pixels")
    print(f"   Erreur moy: {error_stats['mean_error']:.2f} pixels")
    
    # 4. Récupérer la taille de référence
    with rasterio.open(ref_image) as src:
        ref_shape = (src.height, src.width)
    print(f"\n📏 Taille de référence: {ref_shape[1]} x {ref_shape[0]}")
    
    # 5. Appliquer la transformation
    print(f"\n🔄 Application de la transformation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = apply_affine_to_image(
        src_image, 
        out_image, 
        M,
        output_shape=ref_shape,
        order=1
    )
    
    if success:
        # Vérifier le fichier créé
        with rasterio.open(out_image) as src:
            out_shape = (src.height, src.width)
        print(f"   ✅ Image alignée créée: {out_image}")
        print(f"   Taille: {out_shape[1]} x {out_shape[0]}")
    else:
        print(f"   ❌ Échec de l'alignement")
        return
    
    print(f"\n{'='*60}")
    print("TEST NDVI TERMINÉ")
    print(f"{'='*60}")


def test_align_09h_temp_on_13h():
    """
    Test d'alignement de 09h Temp sur 13h NDVI (comme référence spatiale).
    Utilise les fichiers RoiSet_13_9temp_set1.csv (13h NDVI) et set2.csv (09h Temp).
    """
    print("\n" + "=" * 60)
    print("TEST: ALIGNEMENT 09h TEMP -> 13h NDVI")
    print("=" * 60)
    
    # Chemins des fichiers
    rois_dir = os.path.join(WORK_DIR, "Rois_0_align_crop_affine")
    ref_csv = os.path.join(rois_dir, "RoiSet_13_9temp_set1.csv")  # 13h NDVI (référence)
    src_csv = os.path.join(rois_dir, "RoiSet_13_9temp_set2.csv")  # 09h Temp
    
    # Images
    src_image = os.path.join(CROPS_DIR, "09h_Temp_crop.tif")
    ref_image = os.path.join(CROPS_DIR, "13h_NDVI.tif")  # On utilise 13h NDVI comme référence spatiale
    out_image = os.path.join(OUTPUT_DIR, "09h_Temp_aligned.tif")
    
    # Vérifier que les fichiers existent
    for f in [ref_csv, src_csv, src_image, ref_image]:
        if not os.path.exists(f):
            print(f"❌ Fichier manquant: {f}")
            return
    
    # 1. Charger les points
    print(f"\n📍 Chargement des points...")
    pts_ref, pts_src = load_control_points_from_two_files(ref_csv, src_csv)
    print(f"   {len(pts_ref)} points de correspondance")
    
    # Afficher quelques points
    print(f"\n   Exemples de correspondances:")
    for i in range(min(3, len(pts_ref))):
        print(f"      13h NDVI: ({pts_ref[i,0]:.0f}, {pts_ref[i,1]:.0f}) <-> 09h Temp: ({pts_src[i,0]:.0f}, {pts_src[i,1]:.0f})")
    
    # 2. Calculer la transformation
    print(f"\n📐 Calcul de la transformation affine...")
    M = compute_affine_transform(pts_ref, pts_src)
    
    print(f"\n   Matrice de transformation (09h Temp -> 13h NDVI):")
    print(f"   | {M[0,0]:8.5f}  {M[0,1]:8.5f}  {M[0,2]:8.2f} |")
    print(f"   | {M[1,0]:8.5f}  {M[1,1]:8.5f}  {M[1,2]:8.2f} |")
    print(f"   | {M[2,0]:8.5f}  {M[2,1]:8.5f}  {M[2,2]:8.2f} |")
    
    # 3. Calculer l'erreur
    print(f"\n📊 Qualité du recalage:")
    error_stats = compute_transform_error(pts_ref, pts_src, M)
    print(f"   RMSE:       {error_stats['rmse']:.2f} pixels")
    print(f"   Erreur max: {error_stats['max_error']:.2f} pixels")
    print(f"   Erreur moy: {error_stats['mean_error']:.2f} pixels")
    
    # 4. Récupérer la taille de référence (13h NDVI)
    with rasterio.open(ref_image) as src:
        ref_shape = (src.height, src.width)
    print(f"\n📏 Taille de référence (13h NDVI): {ref_shape[1]} x {ref_shape[0]}")
    
    # 5. Appliquer la transformation
    print(f"\n🔄 Application de la transformation...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    success = apply_affine_to_image(
        src_image, 
        out_image, 
        M,
        output_shape=ref_shape,
        order=1
    )
    
    if success:
        # Vérifier le fichier créé
        with rasterio.open(out_image) as src:
            out_shape = (src.height, src.width)
        print(f"   ✅ Image alignée créée: {out_image}")
        print(f"   Taille: {out_shape[1]} x {out_shape[0]}")
    else:
        print(f"   ❌ Échec de l'alignement")
        return
    
    print(f"\n{'='*60}")
    print("TEST TEMP TERMINÉ")
    print(f"{'='*60}")


def main():
    """Point d'entrée principal."""
    # Test 1: 09h NDVI -> 13h NDVI
    test_align_09h_ndvi_on_13h()
    
    # Test 2: 09h Temp -> 13h NDVI
    test_align_09h_temp_on_13h()
    
    print("\n" + "=" * 60)
    print("TOUS LES TESTS TERMINÉS")
    print("=" * 60)
    print(f"\nFichiers créés dans: {OUTPUT_DIR}")
    print("  - 09h_NDVI_aligned.tif")
    print("  - 09h_Temp_aligned.tif")
    print("\nPour vérifier, superposez avec 13h_NDVI.tif dans QGIS ou Fiji")


if __name__ == "__main__":
    main()
