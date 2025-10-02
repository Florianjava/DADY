import tifffile
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset

class TifDataset(Dataset):
    """Dataset pour charger les fichiers .tif avec contrôle par nom et masquage spécifique."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.tif_files = list(self.data_dir.glob("*.tif"))
        self.transform = transform

    def __len__(self):
        return len(self.tif_files)

    def __getitem__(self, idx):
        tif_path = self.tif_files[idx]
        image = tifffile.imread(tif_path)  # (H, W, C)
        image = torch.from_numpy(image).float() / 255.0
        image = image.permute(2, 0, 1)  # (C, H, W)
        if self.transform:
            image = self.transform(image)
        mask_vector = self.create_mask(tif_path.name)
        invalid_mask = self.create_invalid_mask(tif_path.name)
        return image, mask_vector, invalid_mask

    def create_invalid_mask(self, filename):
        invalid = torch.zeros(5)
        name = filename.lower()
        if name.startswith("roujola") or name.startswith("godet"):
            invalid[0] = 1.0
        return invalid

    def create_mask(self, filename):
        mask = torch.zeros(5)
        name = filename.lower()
        if name.startswith("roujola") or name.startswith("godet"):
            mask[0] = 1.0
            second = random.choice([1,2,3,4])
            mask[second] = 1.0
        else:
            for c in random.sample(range(5), random.choice([1,2])):
                mask[c] = 1.0
        return mask
