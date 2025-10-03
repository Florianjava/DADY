from extraction_utils import extract_timeseries

# ==============================
# PARAMÈTRES UTILISATEUR
# ==============================

RASTER_FOLDER = r"D:\Mes Donnees\dady_data_1\data\UC3_riz" # Dossier contenant les images raster
SHP_PATH = r"D:\Mes Donnees\dady_data_1\data\UC3_riz\shapefile_mada.shp" # Chemin vers le shapefile
OUTPUT_DIR = "example" # Dossier de sortie
ID_COLUMN="id" # Colonne dans le shapefile contenant les IDs des parcelles (id pour l'id de la parcelle, band_plot au format Bxx-Pxx pour sa position (voir fichier excel pheno_data_v2))
PARCEL_ID = 1 # ID de la parcelle à extraire
FULL_DATASET = False # Si True, extrait tout le jeu de données d'un coup
PATCH_ONLY = True # Si True, extrait uniquement les patchs (sinon, le polygone et la zone autour))

extract_timeseries(
    raster_folder=RASTER_FOLDER,
    shapefile_path=SHP_PATH,
    output_dir=OUTPUT_DIR,
    id_column=ID_COLUMN,
    parcel_id=PARCEL_ID,
    full_dataset=FULL_DATASET,
    patch_only=PATCH_ONLY
)
