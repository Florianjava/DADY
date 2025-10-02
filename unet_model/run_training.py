from dataloader import TifDataset
from model import ChannelReconstructionUNet
from train_loop import train_model
from torch.utils.data import DataLoader
import torch

train_dataset = TifDataset("224x224_patchs")
val_dataset = TifDataset("224x224_patchs")

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChannelReconstructionUNet()

train_model(model, train_loader, val_loader, num_epochs=3, lr=1e-3, device=device)
