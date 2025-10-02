from extraction_utils import extract_timeseries

# ==============================
# PARAMÈTRES UTILISATEUR
# ==============================

RASTER_FOLDER = r"D:\Mes Donnees\dady_data_1\data\UC2_faidherbia\orthoimages_MS_2021" # Dossier contenant les images raster
SHP_FOLDER = r"D:\Mes Donnees\dady_data_1\data\UC2_faidherbia\shapefile\2021\subplots_2021\subplots_2021" # Chemin vers le shapefile
OUTPUT_DIR = "output" # Dossier de sortie
TREE_ID="P01A1" # parcelle xx et arbre X 
FULL_DATASET = False # Si True, extrait tout le jeu de données d'un coup
PATCH_ONLY = True # Si True, extrait uniquement les patchs (sinon, le polygone et la zone autour))

extract_timeseries(
    raster_folder=RASTER_FOLDER,
    shapefile_folder=SHP_FOLDER,
    tree_id=TREE_ID,
    output_dir=OUTPUT_DIR,
    patch_only=PATCH_ONLY,
    full_dataset=FULL_DATASET
)
