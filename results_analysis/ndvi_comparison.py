import matplotlib.pyplot as plt
import numpy as np
import tifffile

ORIGINAL_IMAGE_PATH = "data_example/224x224_patchs_examples/UC3/andrano1.tif"
RECONSTRUCTED_IMAGE_PATH = "output/reconstructed_image.tif"

def plot_ndvi_comparison(original, predicted, red_channel=2, nir_channel=4, cmap='RdYlGn'):
    """
    Calcule et affiche le NDVI pour l'image originale et l'image reconstruite.
    
    Args:
        original (np.array): Image originale (H, W, C) ou (C, H, W)
        predicted (np.array): Image reconstruite (H, W, C) ou (C, H, W)
        red_channel (int): index du canal rouge
        nir_channel (int): index du canal NIR
        cmap (str): colormap pour NDVI
    """
    # S'assurer que les images ont la forme (H, W, C)
    if original.ndim == 3 and original.shape[0] <= 10:  # Si forme (C, H, W)
        original = np.transpose(original, (1, 2, 0))
    
    if predicted.ndim == 3 and predicted.shape[0] <= 10:  # Si forme (C, H, W)
        predicted = np.transpose(predicted, (1, 2, 0))
    
    # Calcul NDVI
    def compute_ndvi(img):
        nir = img[:, :, nir_channel]
        red = img[:, :, red_channel]
        ndvi = (nir - red) / (nir + red + 1e-8)  # éviter division par 0
        return ndvi

    ndvi_original = compute_ndvi(original)
    ndvi_pred = compute_ndvi(predicted)
    
    # Calculer la différence
    ndvi_diff = ndvi_pred - ndvi_original
    
    # Statistiques
    mae_ndvi = np.mean(np.abs(ndvi_diff))
    rmse_ndvi = np.sqrt(np.mean(ndvi_diff**2))
    corr_ndvi = np.corrcoef(ndvi_original.ravel(), ndvi_pred.ravel())[0, 1]

    # Plot côte à côte avec différence
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # NDVI Original
    im0 = axs[0].imshow(ndvi_original, cmap=cmap, vmin=-1, vmax=1)
    axs[0].set_title(f"NDVI - Image originale\nMoyenne: {np.mean(ndvi_original):.3f}", 
                     fontsize=11)
    axs[0].axis('off')
    fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04, label='NDVI')

    # NDVI Reconstruit
    im1 = axs[1].imshow(ndvi_pred, cmap=cmap, vmin=-1, vmax=1)
    axs[1].set_title(f"NDVI - Image reconstruite\nMoyenne: {np.mean(ndvi_pred):.3f}", 
                     fontsize=11)
    axs[1].axis('off')
    fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04, label='NDVI')
    
    # Différence NDVI
    vmax_diff = max(np.abs(ndvi_diff).max(), 0.1)
    im2 = axs[2].imshow(ndvi_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff)
    axs[2].set_title(f"Différence NDVI (Pred - Original)\nMAE: {mae_ndvi:.4f} | RMSE: {rmse_ndvi:.4f}", 
                     fontsize=11)
    axs[2].axis('off')
    cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04, label='Différence')
    cbar2.ax.axhline(y=0.5, color='black', linewidth=2, linestyle='--')
    
    # Titre global avec corrélation
    fig.suptitle(f"Comparaison NDVI | Corrélation: {corr_ndvi:.4f}", 
                fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher les statistiques dans la console
    print(f"\n{'='*50}")
    print(f"STATISTIQUES NDVI")
    print(f"{'='*50}")
    print(f"MAE (Mean Absolute Error):  {mae_ndvi:.6f}")
    print(f"RMSE (Root Mean Square Error): {rmse_ndvi:.6f}")
    print(f"Corrélation (Pearson):      {corr_ndvi:.6f}")
    print(f"NDVI Original - Min: {ndvi_original.min():.3f}, Max: {ndvi_original.max():.3f}, Moyenne: {np.mean(ndvi_original):.3f}")
    print(f"NDVI Reconstruit - Min: {ndvi_pred.min():.3f}, Max: {ndvi_pred.max():.3f}, Moyenne: {np.mean(ndvi_pred):.3f}")
    print(f"{'='*50}\n")

# Charger les images
original = tifffile.imread(ORIGINAL_IMAGE_PATH).astype(np.float32) / 255.0
reconstructed = tifffile.imread(RECONSTRUCTED_IMAGE_PATH).astype(np.float32)

# Appeler la fonction avec les images chargées
plot_ndvi_comparison(original, reconstructed, red_channel=2, nir_channel=4)