import matplotlib.pyplot as plt
import numpy as np
import tifffile

ORIGINAL_IMAGE_PATH = "data_example/224x224_patchs_examples/UC3/andrano1.tif"
RECONSTRUCTED_IMAGE_PATH = "output/reconstructed_image.tif"
MASK_VECTOR = [0, 0, 0, 0, 1]

def plot_masked_channel_difference(original, predicted, mask_vector, cmap='RdBu_r'):
    """
    Affiche la différence du canal masqué (prédiction - original) avec une palette divergente centrée sur 0.
    
    Args:
        original (np.array): Image originale (H, W, C)
        predicted (np.array): Image prédite (H, W, C) ou (C, H, W)
        mask_vector (list ou np.array): vecteur de masque (1 = masqué)
        cmap (str): colormap divergente (ex: 'RdBu_r', 'seismic')
    """
    # Convertir mask_vector en numpy si c'est une liste
    if isinstance(mask_vector, list):
        mask_vector = np.array(mask_vector)
    
    # Trouver les indices des canaux masqués
    masked_channels = np.where(mask_vector == 1)[0]
    
    # S'assurer que predicted a la forme (H, W, C)
    if predicted.shape[0] == len(mask_vector):  # Si forme (C, H, W)
        predicted = np.transpose(predicted, (1, 2, 0))
    
    # S'assurer que original a la forme (H, W, C)
    if original.shape[0] == len(mask_vector):  # Si forme (C, H, W)
        original = np.transpose(original, (1, 2, 0))
    
    for ch in masked_channels:
        diff = predicted[:, :, ch] - original[:, :, ch]
        
        # Bornes symétriques autour de 0
        vmax = np.max(np.abs(diff)) * 0.6
        vmax = max(vmax, 0.15)  # Au minimum 0.15 pour la visibilité
        vmin = -vmax
        
        # Statistiques pour le titre
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        rmse = np.sqrt(np.mean(diff**2))
        
        plt.figure(figsize=(8, 7))
        im = plt.imshow(diff, cmap=cmap, vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04, label='Différence (Pred - Original)')
        
        # Marquer 0 sur la colorbar
        cbar.ax.axhline(y=0.5, color='black', linewidth=2, linestyle='--')
        
        plt.title(f"Différence du canal masqué {ch} (Reconstruction - Réel)\n" +
                 f"Moyenne: {mean_diff:.4f} | Écart-type: {std_diff:.4f} | RMSE: {rmse:.4f}",
                 fontsize=11)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

# Charger les images
original = tifffile.imread(ORIGINAL_IMAGE_PATH).astype(np.float32) / 255.0
reconstructed = tifffile.imread(RECONSTRUCTED_IMAGE_PATH).astype(np.float32)

# Appeler la fonction avec les images chargées
plot_masked_channel_difference(original, reconstructed, MASK_VECTOR, cmap='RdBu_r')