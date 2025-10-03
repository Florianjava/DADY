import torch
import tifffile
from pathlib import Path
from model import ChannelReconstructionUNet
import numpy as np
import requests

MODEL_PATH = Path("trained_models/unet/channel_prediction_unet.pth")
IMAGE_PATH = "data_example/224x224_patchs_examples/UC3/andrano1.tif"
OUTPUT_DIR = "output"
# URL du modèle sur Hugging Face
HF_URL = "https://huggingface.co/flodel07/missing_channel_prediction/resolve/main/channel_prediction_unet.pth?download=true"

def download_model_if_needed(model_path, url):
    if not model_path.exists():
        print(f"➡️ Téléchargement du modèle depuis {url} ...")
        model_path.write_bytes(requests.get(url).content)
        print("✅ Modèle téléchargé avec succès !")
    else:
        print("✅ Modèle trouvé en local.")

download_model_if_needed(MODEL_PATH, HF_URL)

# --- Config ---
device = torch.device('cpu')
model_path = MODEL_PATH
image_path = IMAGE_PATH

# --- Charger le modèle ---
model = ChannelReconstructionUNet().to(device)
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# --- Lire l'image ---
image = tifffile.imread(image_path)  # (H, W, C)
image_np = image.astype(np.float32) / 255.0  
image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).to(device)  # (1, C, H, W)

# --- Choisir quel canal masquer  ---
mask_vector = [0, 0, 0, 0, 1] 
mask_vector = torch.tensor(mask_vector, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 5)

# --- Appliquer le masque sur l'image d'entrée ---
masked_input = image_tensor.clone()
for c in range(5):
    if mask_vector[0, c] == 1:
        masked_input[0, c] = -1.0

# --- Inférence ---
with torch.no_grad():
    output = model(masked_input, mask_vector)  # (1, C, H, W)
    output_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C)

# --- Calcul RMSE par canal ---
rmse_per_channel = np.sqrt(np.mean((output_np - image_np)**2, axis=(0, 1)))
for c, rmse in enumerate(rmse_per_channel):
    print(f"Canal {c} : RMSE = {rmse:.4f}")


save_dir = Path(OUTPUT_DIR)
save_dir.mkdir(parents=True, exist_ok=True)

print(output_np.shape)
output_np = np.transpose(output_np, (2, 0, 1)).astype(np.float32)
print(output_np.shape)

tifffile.imwrite(
    save_dir / "reconstructed_image.tif",
    output_np.astype(np.float32)
)