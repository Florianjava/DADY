import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

def masked_rmse_loss(predicted, target, invalid_mask_vector):
    batch_size = predicted.shape[0]
    total_loss = torch.tensor(0., device=predicted.device)
    for i in range(batch_size):
        valid_idx = torch.where(invalid_mask_vector[i]==0)[0]
        pred_masked = predicted[i][valid_idx]
        target_masked = target[i][valid_idx]
        mse = F.mse_loss(pred_masked, target_masked, reduction='mean')
        total_loss += torch.sqrt(mse)
    return total_loss / batch_size

def train_model(model, train_loader, val_loader, num_epochs=3, lr=1e-3, device='cuda'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    writer = SummaryWriter(log_dir='runs/channel_reconstruction')

    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.
        for images, mask_vectors, invalid_masks in train_loader:
            images = images.to(device); mask_vectors = mask_vectors.to(device); invalid_masks = invalid_masks.to(device)
            optimizer.zero_grad()
            outputs = model(images, mask_vectors)
            loss = masked_rmse_loss(outputs, images, invalid_masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.
        with torch.no_grad():
            for images, mask_vectors, invalid_masks in val_loader:
                images = images.to(device); mask_vectors = mask_vectors.to(device); invalid_masks = invalid_masks.to(device)
                outputs = model(images, mask_vectors)
                loss = masked_rmse_loss(outputs, images, invalid_masks)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
