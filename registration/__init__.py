"""
registration - Package pour le recalage des données multi-temporelles

Ce package permet de préparer les données pour un recalage,
puis d'appliquer les transformations calculées à tous les canaux.

Workflow:
    1. crop_orthomosaics.py : Crop des orthomosaïques avec coordonnées CSV
    2. align_pairwise.py : Recalage affine par points de correspondance
    3. nonlinear_registration.py : Recalage non-linéaire par champ de déformation

Projet DADY - CIRAD
"""

__version__ = "0.2.0"
__author__ = "DADY Team"

from .prepare_for_fijiyama import (
    prepare_orthomosaics_for_fijiyama,
    compute_mean_ms,
    compute_ndvi_from_bands,
    create_test_dataset
)

from .crop_orthomosaics import (
    crop_all_orthomosaics,
    load_crop_coordinates,
    process_all_crops
)

from .nonlinear_registration import (
    NonLinearTransform,
    compute_all_transforms,
    compose_transforms
)

from .align_all_to_reference import (
    align_all,
    compute_transform,
    load_points
)
