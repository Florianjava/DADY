import matplotlib.pyplot as plt
import numpy as np
import torch
import tifffile

ORIGINAL_IMAGE_PATH = "data_example/224x224_patchs_examples/UC3/andrano1.tif"
RECONSTRUCTED_IMAGE_PATH = "output/reconstructed_image.tif"  # Chemin correct vers l'image reconstruite
MASK_VECTOR = [0, 0, 0, 0, 1]  # Exemple de vecteur de masque

def plot_masked_channel_error_histogram(original, predicted, mask_vector, bins=50):
    """
    Affiche l'histogramme des erreurs pour le(s) canal(aux) masqué(s).

    Args:
        original (np.array): Image originale (H, W, C)
        predicted (np.array): Image prédite (H, W, C) ou (C, H, W)
        mask_vector (list ou torch.Tensor): vecteur de masque (1 = masqué)
        bins (int): nombre de bins pour l'histogramme
    """
    # Convertir mask_vector en numpy si c'est une liste
    if isinstance(mask_vector, list):
        mask_vector = np.array(mask_vector)
    elif isinstance(mask_vector, torch.Tensor):
        mask_vector = mask_vector.cpu().numpy()
    
    # Trouver les indices des canaux masqués
    masked_channels = np.where(mask_vector == 1)[0]
    
    # S'assurer que predicted a la forme (H, W, C)
    if predicted.shape[0] == len(mask_vector):  # Si forme (C, H, W)
        predicted = np.transpose(predicted, (1, 2, 0))
    
    # S'assurer que original a la forme (H, W, C)
    if original.shape[0] == len(mask_vector):  # Si forme (C, H, W)
        original = np.transpose(original, (1, 2, 0))

    for ch in masked_channels:
        error = predicted[:, :, ch] - original[:, :, ch]

        plt.figure(figsize=(8, 5))
        plt.hist(error.ravel(), bins=bins, color='gray', edgecolor='black', alpha=0.7)
        plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur = 0')
        
        # Statistiques
        mean_error = np.mean(error)
        std_error = np.std(error)
        plt.axvline(mean_error, color='blue', linestyle='--', linewidth=2, 
                   label=f'Moyenne = {mean_error:.4f}')
        
        plt.title(f"Histogramme des erreurs - Canal masqué {ch}")
        plt.xlabel("Erreur par pixel (Prédiction - Original)")
        plt.ylabel("Nombre de pixels")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # Ajouter les stats dans le titre
        plt.suptitle(f"Moyenne: {mean_error:.4f} | Écart-type: {std_error:.4f}", 
                    fontsize=10, y=0.98)
        
        plt.tight_layout()
        plt.show()

# Charger les images
original = tifffile.imread(ORIGINAL_IMAGE_PATH).astype(np.float32) / 255.0
reconstructed = tifffile.imread(RECONSTRUCTED_IMAGE_PATH).astype(np.float32)

# Appeler la fonction avec les images chargées
plot_masked_channel_error_histogram(original, reconstructed, MASK_VECTOR, bins=100)