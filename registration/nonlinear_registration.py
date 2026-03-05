#!/usr/bin/env python3
"""
nonlinear_registration.py - Recalage non-linéaire par champ de déformation

Ce module implémente un recalage non-linéaire basé sur des points de correspondance,
en utilisant une interpolation par diffusion gaussienne.

Trois modes de recalage :
    1. Affine seul : transformation affine globale
    2. Non-linéaire seul : champ de déformation par diffusion gaussienne
    3. Affine + Non-linéaire : affine global + corrections locales sur les résidus

Usage:
    from registration.nonlinear_registration import NonLinearTransform
    
    transform = NonLinearTransform()
    transform.compute_from_points(pts_ref, pts_src, mode='affine+nonlinear')
    aligned_image = transform.apply(source_image)
"""

import os
import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates, gaussian_filter
import rasterio
from typing import Tuple, Dict, Optional, Union
import json
import time


# Taille max pour le calcul du champ gaussien (au-delà, on downsample)
MAX_FIELD_SIZE = 1000


class NonLinearTransform:
    """
    Classe pour le recalage non-linéaire par champ de déformation.
    
    Encapsule :
        - Transformation affine (matrice 3x3)
        - Champ de déformation résiduel (field_x, field_y)
        - Champ de déformation total combiné
    
    Attributes:
        affine_matrix: Matrice affine 3x3 (None si non calculée)
        residual_field_x: Champ de déplacement X résiduel (None si non calculé)
        residual_field_y: Champ de déplacement Y résiduel (None si non calculé)
        output_shape: Taille de l'image de sortie (height, width)
        sigma: Paramètre sigma du filtre gaussien
        mode: Mode de transformation ('affine', 'nonlinear', 'affine+nonlinear')
    """
    
    def __init__(self):
        """Initialise une transformation vide."""
        self.affine_matrix: Optional[np.ndarray] = None
        self.residual_field_x: Optional[np.ndarray] = None
        self.residual_field_y: Optional[np.ndarray] = None
        self.output_shape: Optional[Tuple[int, int]] = None
        self.sigma: float = 0
        self.mode: str = 'affine'
        self._pts_ref: Optional[np.ndarray] = None
        self._pts_src: Optional[np.ndarray] = None
        self._error_stats: Optional[Dict] = None
    
    # =========================================================================
    # CALCUL DES TRANSFORMATIONS
    # =========================================================================
    
    def compute_from_points(
        self,
        pts_ref: np.ndarray,
        pts_src: np.ndarray,
        output_shape: Tuple[int, int],
        mode: str = 'affine+nonlinear',
        sigma: float = 0
    ) -> 'NonLinearTransform':
        """
        Calcule la transformation à partir de points de correspondance.
        
        Args:
            pts_ref: Points de référence (N, 2) - coordonnées (x, y)
            pts_src: Points source (N, 2) - coordonnées (x, y)
            output_shape: Taille de sortie (height, width)
            mode: 'affine', 'nonlinear', ou 'affine+nonlinear'
            sigma: Paramètre du filtre gaussien (0 = auto = 1/15 de la taille)
            
        Returns:
            self (pour chaînage)
        """
        self._pts_ref = pts_ref.copy()
        self._pts_src = pts_src.copy()
        self.output_shape = output_shape
        self.mode = mode
        
        # Calcul du sigma
        if sigma == 0:
            # Auto : 1/15 de la dimension moyenne
            self.sigma = (output_shape[0] + output_shape[1]) / 2 / 15
        else:
            self.sigma = sigma
        
        print(f"   Sigma gaussien: {self.sigma:.1f} pixels")
        
        if mode == 'affine':
            self._compute_affine(pts_ref, pts_src)
            
        elif mode == 'nonlinear':
            self._compute_nonlinear_field(pts_ref, pts_src, output_shape)
            
        elif mode == 'affine+nonlinear':
            # 1. Calculer l'affine : M tel que pts_ref = M @ pts_src
            self._compute_affine(pts_ref, pts_src)
            
            # 2. Calculer où M_inv nous envoie depuis pts_ref
            # M_inv @ pts_ref devrait nous ramener à pts_src si M était parfait
            M_inv = np.linalg.inv(self.affine_matrix)
            pts_ref_h = np.hstack([pts_ref, np.ones((pts_ref.shape[0], 1))])
            pts_via_inv = (M_inv @ pts_ref_h.T).T[:, :2]
            
            # 3. Calculer le résidu dans l'ESPACE SOURCE
            # pts_via_inv est où M_inv nous envoie
            # pts_src est où on VEUT aller
            # résidu = pts_src - pts_via_inv = correction à ajouter au champ
            residuals_x = pts_src[:, 0] - pts_via_inv[:, 0]
            residuals_y = pts_src[:, 1] - pts_via_inv[:, 1]
            
            print(f"   [Résidus dans espace SRC] Moyenne: dx={np.mean(residuals_x):.2f}, dy={np.mean(residuals_y):.2f}")
            print(f"   [Résidus dans espace SRC] Écart-type: dx={np.std(residuals_x):.2f}, dy={np.std(residuals_y):.2f}")
            
            # 4. Construire le champ de déformation
            # Le champ est indexé par les positions dans REF (pts_ref)
            # mais les valeurs sont les corrections dans l'espace SRC
            self._compute_residual_field(pts_ref, residuals_x, residuals_y, output_shape)
            
        else:
            raise ValueError(f"Mode inconnu: {mode}. Utiliser 'affine', 'nonlinear', ou 'affine+nonlinear'")
        
        # Calculer les statistiques d'erreur
        self._compute_error_stats()
        
        return self
    
    def _compute_affine(self, pts_ref: np.ndarray, pts_src: np.ndarray) -> None:
        """
        Calcule la transformation affine par moindres carrés.
        
        On cherche M tel que : pts_ref = M @ pts_src
        Autrement dit : pour un point (x_src, y_src) dans l'image source,
        M donne sa position (x_ref, y_ref) dans l'image de référence.
        
        Pour appliquer la transformation (aller chercher les valeurs dans src
        pour remplir ref), on utilisera M_inv.
        """
        n_points = pts_ref.shape[0]
        
        if n_points < 3:
            raise ValueError("Il faut au moins 3 points pour une transformation affine")
        
        # Système linéaire : pts_ref = M @ pts_src
        # [x_ref]   [a b c] [x_src]
        # [y_ref] = [d e f] [y_src]
        # [  1  ]   [0 0 1] [  1  ]
        A = np.zeros((2 * n_points, 6))
        b = np.zeros(2 * n_points)
        
        for i in range(n_points):
            x, y = pts_src[i]
            x_ref, y_ref = pts_ref[i]
            
            A[2*i, 0] = x
            A[2*i, 1] = y
            A[2*i, 2] = 1
            b[2*i] = x_ref
            
            A[2*i+1, 3] = x
            A[2*i+1, 4] = y
            A[2*i+1, 5] = 1
            b[2*i+1] = y_ref
        
        params, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        self.affine_matrix = np.array([
            [params[0], params[1], params[2]],
            [params[3], params[4], params[5]],
            [0, 0, 1]
        ])
        
        # Vérification : appliquer M aux points src doit donner pts_ref
        pts_check = self._apply_affine_to_points(pts_src)
        err = np.sqrt(np.mean((pts_check - pts_ref)**2))
        print(f"   [Affine] Vérification RMSE (M @ pts_src vs pts_ref): {err:.4f} px")
    
    def _apply_affine_to_points(self, pts: np.ndarray) -> np.ndarray:
        """Applique la transformation affine à un ensemble de points."""
        if self.affine_matrix is None:
            return pts.copy()
        
        n = pts.shape[0]
        pts_h = np.hstack([pts, np.ones((n, 1))])
        pts_transformed = (self.affine_matrix @ pts_h.T).T
        return pts_transformed[:, :2]
    
    def _compute_nonlinear_field(
        self,
        pts_ref: np.ndarray,
        pts_src: np.ndarray,
        output_shape: Tuple[int, int]
    ) -> None:
        """
        Calcule le champ de déformation non-linéaire par diffusion gaussienne.
        
        Le champ est stocké directement dans residual_field_x/y (pas de partie affine).
        """
        # Déplacements aux points de correspondance
        dx = pts_src[:, 0] - pts_ref[:, 0]
        dy = pts_src[:, 1] - pts_ref[:, 1]
        
        self._compute_residual_field(pts_ref, -dx, -dy, output_shape)
    
    def _compute_residual_field(
        self,
        pts_ref: np.ndarray,
        residuals_x: np.ndarray,
        residuals_y: np.ndarray,
        output_shape: Tuple[int, int]
    ) -> None:
        """
        Construit le champ de déformation par diffusion gaussienne.
        
        Méthode :
            1. Si image trop grande, downsampler les coordonnées
            2. Créer des images sparse avec les résidus aux points
            3. Appliquer un filtre gaussien pour diffuser
            4. Normaliser par l'indicatrice lissée
            5. Si downsamplé, upsampler le résultat
        
        Args:
            pts_ref: Positions des points dans l'image de référence (x, y)
            residuals_x: Résidus en X à chaque point
            residuals_y: Résidus en Y à chaque point
            output_shape: Taille (height, width)
        """
        height, width = output_shape
        
        print(f"   [_compute_residual_field] Début - shape={output_shape}")
        t0 = time.time()
        
        # Vérifier si downsampling nécessaire
        max_dim = max(height, width)
        if max_dim > MAX_FIELD_SIZE:
            scale = MAX_FIELD_SIZE / max_dim
            work_height = int(height * scale)
            work_width = int(width * scale)
            print(f"   [_compute_residual_field] Downsampling {width}x{height} -> {work_width}x{work_height} (scale={scale:.3f})")
            
            # Adapter les coordonnées des points et les résidus
            pts_ref_scaled = pts_ref * scale
            residuals_x_scaled = residuals_x * scale
            residuals_y_scaled = residuals_y * scale
            sigma_scaled = self.sigma * scale
        else:
            scale = 1.0
            work_height, work_width = height, width
            pts_ref_scaled = pts_ref
            residuals_x_scaled = residuals_x
            residuals_y_scaled = residuals_y
            sigma_scaled = self.sigma
        
        print(f"   [_compute_residual_field] Taille de travail: {work_width}x{work_height}, sigma={sigma_scaled:.1f}")
        t1 = time.time()
        
        # Images sparse à la résolution de travail
        Dx = np.zeros((work_height, work_width), dtype=np.float64)
        Dy = np.zeros((work_height, work_width), dtype=np.float64)
        W = np.zeros((work_height, work_width), dtype=np.float64)
        
        # Remplir aux points de correspondance
        for i, (x, y) in enumerate(pts_ref_scaled):
            row, col = int(round(y)), int(round(x))
            
            if 0 <= row < work_height and 0 <= col < work_width:
                Dx[row, col] = residuals_x_scaled[i]
                Dy[row, col] = residuals_y_scaled[i]
                W[row, col] = 1.0
        
        print(f"   [_compute_residual_field] Préparation: {time.time()-t1:.2f}s")
        t2 = time.time()
        
        # Diffusion gaussienne (à basse résolution si downsamplé)
        print(f"   [_compute_residual_field] Lissage gaussien (3 arrays)...")
        Dx_smooth = gaussian_filter(Dx, sigma=sigma_scaled, mode='constant', cval=0)
        Dy_smooth = gaussian_filter(Dy, sigma=sigma_scaled, mode='constant', cval=0)
        W_smooth = gaussian_filter(W, sigma=sigma_scaled, mode='constant', cval=0)
        print(f"   [_compute_residual_field] Lissage: {time.time()-t2:.2f}s")
        t3 = time.time()
        
        # Normalisation
        epsilon = 1e-10
        W_smooth_safe = np.maximum(W_smooth, epsilon)
        field_x_work = Dx_smooth / W_smooth_safe
        field_y_work = Dy_smooth / W_smooth_safe
        
        # Upsampling si nécessaire
        if scale < 1.0:
            print(f"   [_compute_residual_field] Upsampling vers {width}x{height}...")
            # Utiliser zoom pour upsampler avec interpolation bilinéaire
            zoom_factor = (height / work_height, width / work_width)
            self.residual_field_x = ndimage.zoom(field_x_work, zoom_factor, order=1) / scale
            self.residual_field_y = ndimage.zoom(field_y_work, zoom_factor, order=1) / scale
            print(f"   [_compute_residual_field] Upsampling: {time.time()-t3:.2f}s")
        else:
            self.residual_field_x = field_x_work
            self.residual_field_y = field_y_work
        
        print(f"   [_compute_residual_field] TOTAL: {time.time()-t0:.2f}s")
    
    def _compute_error_stats(self) -> None:
        """Calcule les statistiques d'erreur sur les points de correspondance."""
        if self._pts_ref is None or self._pts_src is None:
            return
        
        # Transformer les points source
        pts_transformed = self.transform_points(self._pts_src)
        
        # Erreurs par point
        errors = np.sqrt(
            (pts_transformed[:, 0] - self._pts_ref[:, 0])**2 +
            (pts_transformed[:, 1] - self._pts_ref[:, 1])**2
        )
        
        self._error_stats = {
            'rmse': np.sqrt(np.mean(errors**2)),
            'max_error': np.max(errors),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'errors_per_point': errors
        }
    
    # =========================================================================
    # APPLICATION DES TRANSFORMATIONS
    # =========================================================================
    
    def get_combined_field(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retourne le champ de déformation total (affine + résiduel combinés).
        
        Le champ retourné permet de transformer directement avec map_coordinates :
            output[row, col] = input[field_y[row, col], field_x[row, col]]
        
        Returns:
            (field_x, field_y) : Champs de coordonnées pour map_coordinates
        """
        if self.output_shape is None:
            raise ValueError("La transformation n'a pas été calculée")
        
        height, width = self.output_shape
        
        # Grille de coordonnées de l'image de sortie
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        
        if self.mode == 'affine':
            # Transformation affine seule
            if self.affine_matrix is None:
                return cols.astype(np.float64), rows.astype(np.float64)
            
            M_inv = np.linalg.inv(self.affine_matrix)
            
            # Coordonnées dans l'image source
            field_x = M_inv[0, 0] * cols + M_inv[0, 1] * rows + M_inv[0, 2]
            field_y = M_inv[1, 0] * cols + M_inv[1, 1] * rows + M_inv[1, 2]
            
        elif self.mode == 'nonlinear':
            # Champ non-linéaire seul
            if self.residual_field_x is None:
                return cols.astype(np.float64), rows.astype(np.float64)
            
            # Le champ résiduel contient le déplacement src->ref
            # Pour remplir ref avec les valeurs de src, on ajoute le résidu
            field_x = cols + self.residual_field_x
            field_y = rows + self.residual_field_y
            
        elif self.mode == 'affine+nonlinear':
            # Affine puis correction résiduelle
            if self.affine_matrix is None:
                field_x = cols.astype(np.float64)
                field_y = rows.astype(np.float64)
            else:
                M_inv = np.linalg.inv(self.affine_matrix)
                field_x = M_inv[0, 0] * cols + M_inv[0, 1] * rows + M_inv[0, 2]
                field_y = M_inv[1, 0] * cols + M_inv[1, 1] * rows + M_inv[1, 2]
            
            # Ajouter la correction résiduelle
            if self.residual_field_x is not None:
                # Le champ résiduel contient la correction dans l'espace SOURCE
                # residual = pts_src - M_inv(pts_ref)
                # Donc field_corrigé = M_inv(ref) + residual = pts_src (ce qu'on veut)
                field_x = field_x + self.residual_field_x
                field_y = field_y + self.residual_field_y
        
        else:
            field_x = cols.astype(np.float64)
            field_y = rows.astype(np.float64)
        
        return field_x, field_y
    
    def transform_points(self, pts: np.ndarray) -> np.ndarray:
        """
        Transforme un ensemble de points.
        
        Args:
            pts: Points (N, 2) en coordonnées (x, y)
            
        Returns:
            Points transformés (N, 2)
        """
        pts_out = pts.copy()
        
        # 1. Appliquer l'affine
        if self.affine_matrix is not None and self.mode in ['affine', 'affine+nonlinear']:
            pts_out = self._apply_affine_to_points(pts_out)
        
        # 2. Appliquer la correction résiduelle
        # Le champ résiduel donne le déplacement de src vers ref
        # Donc pour transformer un point src -> ref, on AJOUTE le résidu
        if self.residual_field_x is not None and self.mode in ['nonlinear', 'affine+nonlinear']:
            for i, (x, y) in enumerate(pts_out):
                row, col = int(round(y)), int(round(x))
                if 0 <= row < self.residual_field_x.shape[0] and 0 <= col < self.residual_field_x.shape[1]:
                    pts_out[i, 0] += self.residual_field_x[row, col]
                    pts_out[i, 1] += self.residual_field_y[row, col]
        
        return pts_out
    
    def apply(
        self,
        image: np.ndarray,
        order: int = 1,
        cval: float = 0
    ) -> np.ndarray:
        """
        Applique la transformation à une image.
        
        Args:
            image: Image source (2D ou 3D avec bandes en première dimension)
            order: Ordre d'interpolation (0=nearest, 1=bilinear, 3=cubic)
            cval: Valeur pour les pixels hors image
            
        Returns:
            Image transformée
        """
        field_x, field_y = self.get_combined_field()
        
        # Gérer les images multi-bandes
        if image.ndim == 2:
            return map_coordinates(
                image,
                [field_y, field_x],
                order=order,
                mode='constant',
                cval=cval
            )
        elif image.ndim == 3:
            # Bandes en première dimension
            output = np.zeros((image.shape[0], self.output_shape[0], self.output_shape[1]), 
                             dtype=image.dtype)
            for b in range(image.shape[0]):
                output[b] = map_coordinates(
                    image[b],
                    [field_y, field_x],
                    order=order,
                    mode='constant',
                    cval=cval
                )
            return output
        else:
            raise ValueError(f"Image doit être 2D ou 3D, pas {image.ndim}D")
    
    def apply_to_file(
        self,
        src_path: str,
        dst_path: str,
        ref_profile: Dict = None,
        order: int = 1
    ) -> bool:
        """
        Applique la transformation à un fichier GeoTIFF.
        
        Args:
            src_path: Chemin de l'image source
            dst_path: Chemin de sortie
            ref_profile: Profil rasterio de référence (pour géoréférencement)
            order: Ordre d'interpolation
            
        Returns:
            True si succès
        """
        try:
            with rasterio.open(src_path) as src:
                data = src.read()
                profile = src.profile.copy()
            
            # Appliquer la transformation
            output = self.apply(data, order=order)
            
            # Mettre à jour le profil
            if ref_profile is not None:
                profile.update(
                    transform=ref_profile.get('transform', profile['transform']),
                    crs=ref_profile.get('crs', profile['crs']),
                    height=self.output_shape[0],
                    width=self.output_shape[1]
                )
            else:
                profile.update(
                    height=self.output_shape[0],
                    width=self.output_shape[1]
                )
            
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with rasterio.open(dst_path, 'w', **profile) as dst:
                dst.write(output)
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # =========================================================================
    # SAUVEGARDE / CHARGEMENT
    # =========================================================================
    
    def save(self, path: str) -> None:
        """
        Sauvegarde la transformation dans un fichier.
        
        Sauvegarde :
            - {path}_affine.npy : matrice affine
            - {path}_field_x.npy : champ X résiduel
            - {path}_field_y.npy : champ Y résiduel
            - {path}_info.json : métadonnées
        """
        base = path.rstrip('.npy').rstrip('.json')
        
        # Sauvegarder les composants
        if self.affine_matrix is not None:
            np.save(f"{base}_affine.npy", self.affine_matrix)
        
        if self.residual_field_x is not None:
            np.save(f"{base}_field_x.npy", self.residual_field_x)
            np.save(f"{base}_field_y.npy", self.residual_field_y)
        
        # Métadonnées
        info = {
            'mode': self.mode,
            'sigma': self.sigma,
            'output_shape': self.output_shape,
            'has_affine': self.affine_matrix is not None,
            'has_field': self.residual_field_x is not None,
            'error_stats': {
                k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in (self._error_stats or {}).items()
                if k != 'errors_per_point'
            }
        }
        
        with open(f"{base}_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"💾 Transformation sauvegardée: {base}_*.npy/json")
    
    @classmethod
    def load(cls, path: str) -> 'NonLinearTransform':
        """
        Charge une transformation depuis un fichier.
        
        Args:
            path: Chemin de base (sans suffixe)
            
        Returns:
            Instance de NonLinearTransform
        """
        base = path.rstrip('.npy').rstrip('.json')
        
        transform = cls()
        
        # Charger les métadonnées
        with open(f"{base}_info.json", 'r') as f:
            info = json.load(f)
        
        transform.mode = info['mode']
        transform.sigma = info['sigma']
        transform.output_shape = tuple(info['output_shape'])
        
        # Charger les composants
        if info['has_affine']:
            transform.affine_matrix = np.load(f"{base}_affine.npy")
        
        if info['has_field']:
            transform.residual_field_x = np.load(f"{base}_field_x.npy")
            transform.residual_field_y = np.load(f"{base}_field_y.npy")
        
        return transform
    
    # =========================================================================
    # UTILITAIRES
    # =========================================================================
    
    def get_error_stats(self) -> Dict:
        """Retourne les statistiques d'erreur."""
        return self._error_stats or {}
    
    def print_summary(self) -> None:
        """Affiche un résumé de la transformation."""
        print(f"\n📐 Transformation {self.mode}")
        print(f"   Taille de sortie: {self.output_shape[1]} x {self.output_shape[0]}")
        print(f"   Sigma: {self.sigma:.1f} pixels")
        
        if self.affine_matrix is not None:
            print(f"\n   Matrice affine:")
            print(f"   | {self.affine_matrix[0,0]:8.5f}  {self.affine_matrix[0,1]:8.5f}  {self.affine_matrix[0,2]:8.2f} |")
            print(f"   | {self.affine_matrix[1,0]:8.5f}  {self.affine_matrix[1,1]:8.5f}  {self.affine_matrix[1,2]:8.2f} |")
        
        if self._error_stats:
            print(f"\n   📊 Erreur de recalage:")
            print(f"      RMSE:       {self._error_stats['rmse']:.2f} pixels")
            print(f"      Erreur max: {self._error_stats['max_error']:.2f} pixels")
            print(f"      Erreur moy: {self._error_stats['mean_error']:.2f} pixels")


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def compose_transforms(
    transform_first: NonLinearTransform,
    transform_second: NonLinearTransform,
    output_shape: Optional[Tuple[int, int]] = None,
    sigma: float = 0
) -> NonLinearTransform:
    """
    Compose deux transformations : result = second ∘ first
    
    C'est-à-dire : on applique d'abord first, puis second.
    
    Exemple d'usage :
        - transform_first : 9h-Temp → 13h-Temp
        - transform_second : 13h-Temp → 13h-NDVI
        - result : 9h-Temp → 13h-NDVI
    
    Pour les parties affines, on compose les matrices : M_result = M_second @ M_first
    Pour les parties non-linéaires, on compose les champs de déformation.
    
    Args:
        transform_first: Première transformation (appliquée en premier)
        transform_second: Deuxième transformation (appliquée ensuite)
        output_shape: Taille de sortie (si None, prend celle de transform_second)
        sigma: Sigma pour le champ composé (0 = auto)
    
    Returns:
        NonLinearTransform composée
    """
    result = NonLinearTransform()
    
    # Taille de sortie = celle de la transformation finale
    if output_shape is None:
        output_shape = transform_second.output_shape
    result.output_shape = output_shape
    
    # Mode : on prend le mode le plus "complet"
    if 'nonlinear' in transform_first.mode or 'nonlinear' in transform_second.mode:
        result.mode = 'affine+nonlinear'
    else:
        result.mode = 'affine'
    
    # Sigma
    if sigma == 0:
        result.sigma = (output_shape[0] + output_shape[1]) / 2 / 15
    else:
        result.sigma = sigma
    
    print(f"\n🔗 Composition de transformations")
    print(f"   First: {transform_first.mode}, Second: {transform_second.mode}")
    print(f"   Résultat: {result.mode}")
    
    # =========================================================================
    # COMPOSITION DES MATRICES AFFINES
    # =========================================================================
    # M_first : src1 → dst1 (= src2)
    # M_second : src2 → dst2
    # M_composed : src1 → dst2 = M_second @ M_first
    
    M_first = transform_first.affine_matrix
    M_second = transform_second.affine_matrix
    
    if M_first is not None and M_second is not None:
        result.affine_matrix = M_second @ M_first
        print(f"   Matrices affines composées: M_result = M_second @ M_first")
    elif M_second is not None:
        result.affine_matrix = M_second.copy()
    elif M_first is not None:
        result.affine_matrix = M_first.copy()
    else:
        result.affine_matrix = np.eye(3)
    
    # =========================================================================
    # COMPOSITION DES CHAMPS NON-LINÉAIRES
    # =========================================================================
    # C'est plus délicat : on doit composer les champs de déformation
    # 
    # Pour chaque pixel (x, y) dans l'image de sortie finale :
    #   1. M_second_inv(x, y) + field_second(x, y) → position dans l'espace intermédiaire
    #   2. M_first_inv(pos_inter) + field_first(pos_inter) → position dans l'image source
    #
    # On construit donc le champ composé directement
    
    height, width = output_shape
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    
    # Étape 1 : Appliquer la transformation inverse de second
    # pour obtenir les coordonnées dans l'espace intermédiaire
    M_second_inv = np.linalg.inv(result.affine_matrix if M_second is None else M_second)
    
    inter_x = M_second_inv[0, 0] * cols + M_second_inv[0, 1] * rows + M_second_inv[0, 2]
    inter_y = M_second_inv[1, 0] * cols + M_second_inv[1, 1] * rows + M_second_inv[1, 2]
    
    # Ajouter le champ résiduel de second (s'il existe)
    if transform_second.residual_field_x is not None:
        inter_x = inter_x + transform_second.residual_field_x
        inter_y = inter_y + transform_second.residual_field_y
    
    # Étape 2 : Depuis l'espace intermédiaire, appliquer la transformation inverse de first
    M_first_inv = np.linalg.inv(np.eye(3) if M_first is None else M_first)
    
    src_x = M_first_inv[0, 0] * inter_x + M_first_inv[0, 1] * inter_y + M_first_inv[0, 2]
    src_y = M_first_inv[1, 0] * inter_x + M_first_inv[1, 1] * inter_y + M_first_inv[1, 2]
    
    # Ajouter le champ résiduel de first (s'il existe)
    # Il faut l'interpoler aux positions intermédiaires
    if transform_first.residual_field_x is not None:
        # Interpoler le champ de first aux positions intermédiaires
        field_first_x_interp = map_coordinates(
            transform_first.residual_field_x,
            [inter_y, inter_x],  # row, col
            order=1,
            mode='constant',
            cval=0
        )
        field_first_y_interp = map_coordinates(
            transform_first.residual_field_y,
            [inter_y, inter_x],
            order=1,
            mode='constant',
            cval=0
        )
        src_x = src_x + field_first_x_interp
        src_y = src_y + field_first_y_interp
    
    # Maintenant on a le champ total : (cols, rows) → (src_x, src_y)
    # On doit le décomposer en affine + résidu
    
    # Le champ affine composé est M_composed_inv @ (cols, rows)
    M_composed_inv = np.linalg.inv(result.affine_matrix)
    affine_x = M_composed_inv[0, 0] * cols + M_composed_inv[0, 1] * rows + M_composed_inv[0, 2]
    affine_y = M_composed_inv[1, 0] * cols + M_composed_inv[1, 1] * rows + M_composed_inv[1, 2]
    
    # Le résidu est la différence
    result.residual_field_x = src_x - affine_x
    result.residual_field_y = src_y - affine_y
    
    # Statistiques sur le résidu composé
    residual_mag = np.sqrt(result.residual_field_x**2 + result.residual_field_y**2)
    print(f"   Champ résiduel composé:")
    print(f"      Magnitude moyenne: {np.mean(residual_mag):.2f} px")
    print(f"      Magnitude max: {np.max(residual_mag):.2f} px")
    
    return result


def compute_all_transforms(
    pts_ref: np.ndarray,
    pts_src: np.ndarray,
    output_shape: Tuple[int, int],
    sigma: float = 0
) -> Dict[str, NonLinearTransform]:
    """
    Calcule les trois types de transformation pour comparaison.
    
    Args:
        pts_ref: Points de référence (N, 2)
        pts_src: Points source (N, 2)
        output_shape: Taille (height, width)
        sigma: Paramètre gaussien (0 = auto)
        
    Returns:
        Dict avec 'affine', 'nonlinear', 'affine+nonlinear'
    """
    transforms = {}
    
    for mode in ['affine', 'nonlinear', 'affine+nonlinear']:
        print(f"\n{'='*40}")
        print(f"Mode: {mode}")
        print(f"{'='*40}")
        
        t = NonLinearTransform()
        t.compute_from_points(pts_ref, pts_src, output_shape, mode=mode, sigma=sigma)
        t.print_summary()
        transforms[mode] = t
    
    return transforms


# =============================================================================
# TEST
# =============================================================================

def test_nonlinear_registration(hour: str = "13", modality: str = "temp"):
    """
    Test du recalage non-linéaire.
    
    La RÉFÉRENCE est TOUJOURS 13h NDVI.
    L'image FLOTTANTE (moving) est spécifiée par hour et modality.
    
    Args:
        hour: Heure de l'image flottante ("09", "11", "13", "15", "17")
        modality: Type de l'image flottante ("ndvi" ou "temp")
    """
    import pandas as pd
    
    modality = modality.lower()
    
    # Référence = TOUJOURS 13h NDVI
    ref_name = "13h_NDVI.tif"
    
    # Image flottante (moving)
    if modality == "ndvi":
        src_name = f"{hour}h_NDVI.tif"
    elif modality == "temp":
        src_name = f"{hour}h_Temp_crop.tif"
    else:
        raise ValueError(f"modality doit être 'ndvi' ou 'temp', pas '{modality}'")
    
    # Noms de sortie
    dst_base = f"{hour}h_{modality.upper()}"
    dst_affine = f"{dst_base}_affine.tif"
    dst_nonlinear = f"{dst_base}_nonlinear.tif"
    
    # Fichiers ROI : convention RoiSet_13_{hour}{modality}_set1/set2.csv
    # set1 = référence (13h NDVI), set2 = image flottante
    roi_prefix = f"RoiSet_13_{hour}{modality}"
    transform_name = f"transform_{hour}h_{modality}_to_13h"
    
    print("=" * 60)
    print(f"RECALAGE: {hour}h {modality.upper()} -> 13h NDVI (référence)")
    print("=" * 60)
    
    # Chemins
    work_dir = "/Donnees/DD_CIRS626_DATA/Test_anatomics_tmp/processed/registration"
    crops_dir = os.path.join(work_dir, "crops_2021")
    rois_dir = os.path.join(work_dir, "Rois_0_align_crop_affine")
    output_dir = os.path.join(work_dir, "aligned_2021")
    
    # Charger les points de correspondance
    # set1 = points dans la RÉFÉRENCE (13h NDVI)
    # set2 = points dans l'image FLOTTANTE (moving)
    ref_csv = os.path.join(rois_dir, f"{roi_prefix}_set1.csv")
    src_csv = os.path.join(rois_dir, f"{roi_prefix}_set2.csv")
    
    print(f"\n📁 Fichiers ROI:")
    print(f"   Référence (13h NDVI): {ref_csv}")
    print(f"   Flottante ({hour}h {modality}): {src_csv}")
    
    pts_ref = pd.read_csv(ref_csv)[['x', 'y']].values
    pts_src = pd.read_csv(src_csv)[['x', 'y']].values
    
    print(f"\n📍 {len(pts_ref)} points de correspondance")
    print(f"   Points ref (dans 13h NDVI) - premier: ({pts_ref[0,0]:.1f}, {pts_ref[0,1]:.1f})")
    print(f"   Points src (dans {hour}h {modality}) - premier: ({pts_src[0,0]:.1f}, {pts_src[0,1]:.1f})")
    
    # Charger l'image de référence (13h NDVI) pour la taille et le profil
    ref_image = os.path.join(crops_dir, ref_name)
    with rasterio.open(ref_image) as src:
        output_shape = (src.height, src.width)
        ref_profile = src.profile.copy()
    
    print(f"\n📏 Taille de référence (13h NDVI): {output_shape[1]} x {output_shape[0]}")
    
    # Calculer les transformations
    transforms = compute_all_transforms(pts_ref, pts_src, output_shape, sigma=0)
    
    # Chemin de l'image source (flottante)
    src_image = os.path.join(crops_dir, src_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # === Appliquer AFFINE seul ===
    print(f"\n{'='*60}")
    print("APPLICATION DE AFFINE SEUL")
    print("=" * 60)
    
    dst_path_affine = os.path.join(output_dir, dst_affine)
    success_affine = transforms['affine'].apply_to_file(
        src_image, dst_path_affine, ref_profile
    )
    if success_affine:
        print(f"✅ Image créée: {dst_path_affine}")
    
    # === Appliquer AFFINE+NONLINEAR ===
    print(f"\n{'='*60}")
    print("APPLICATION DE AFFINE+NONLINEAR")
    print("=" * 60)
    
    dst_path_nonlinear = os.path.join(output_dir, dst_nonlinear)
    success_nonlinear = transforms['affine+nonlinear'].apply_to_file(
        src_image, dst_path_nonlinear, ref_profile
    )
    if success_nonlinear:
        print(f"✅ Image créée: {dst_path_nonlinear}")
    
    # Sauvegarder la transformation affine+nonlinear
    transform_path = os.path.join(output_dir, transform_name)
    transforms['affine+nonlinear'].save(transform_path)
    
    # === Comparaison des erreurs ===
    print(f"\n{'='*60}")
    print("COMPARAISON DES ERREURS (en pixels)")
    print("=" * 60)
    print(f"{'Mode':<20} {'RMSE':>10} {'Max':>10} {'Mean':>10}")
    print("-" * 50)
    for mode, t in transforms.items():
        stats = t.get_error_stats()
        print(f"{mode:<20} {stats['rmse']:>10.2f} {stats['max_error']:>10.2f} {stats['mean_error']:>10.2f}")
    
    print(f"\n{'='*60}")
    print("RÉSUMÉ")
    print("=" * 60)
    print(f"📁 Images créées:")
    print(f"   Affine:           {dst_path_affine}")
    print(f"   Affine+Nonlinear: {dst_path_nonlinear}")


if __name__ == "__main__":
    import sys
    # Usage: python nonlinear_registration.py [hour] [modality]
    # Exemples:
    #   python nonlinear_registration.py 09 ndvi   -> recale 09h NDVI sur 13h NDVI
    #   python nonlinear_registration.py 13 temp   -> recale 13h Temp sur 13h NDVI
    #   python nonlinear_registration.py 09 temp   -> recale 09h Temp sur 13h NDVI
    
    hour = sys.argv[1] if len(sys.argv) > 1 else "13"
    modality = sys.argv[2] if len(sys.argv) > 2 else "temp"
    
    print(f"\n🔧 Paramètres: hour={hour}, modality={modality}")
    test_nonlinear_registration(hour, modality)

