import os
import torch
from torch.utils.data import DataLoader, Subset
from src.dataset import BiTemporalChangeDataset
from src.model import SiameseUNet
from src.metrics import BCEDiceLoss, calculate_iou

def fast_train():
    print("--- STARTING FAST-TRACK MODULE 5 TRAINING ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware device: {device}")

    # Load Datasets
    full_train = BiTemporalChangeDataset("data/dataset_split/train", img_size=(256, 256))
    full_val = BiTemporalChangeDataset("data/dataset_split/validation", img_size=(256, 256))

    # Fast subset (80 training samples, 20 validation samples)
    train_subset = Subset(full_train, range(min(80, len(full_train))))
    val_subset = Subset(full_val, range(min(20, len(full_val))))

    train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

    model = SiameseUNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val_iou = 0.0

    for epoch in range(1, 3):
        model.train()
        train_loss, train_iou = 0.0, 0.0

        for batch_idx, (before, after, masks) in enumerate(train_loader, 1):
            before, after, masks = before.to(device), after.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(before, after)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_iou = calculate_iou(outputs, masks)
            train_loss += loss.item()
            train_iou += batch_iou
            print(f"  [Epoch {epoch}/2] Step {batch_idx}/{len(train_loader)} | Batch Loss: {loss.item():.4f} | Batch IoU: {batch_iou:.4f}", flush=True)

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for before, after, masks in val_loader:
                before, after, masks = before.to(device), after.to(device), masks.to(device)
                outputs = model(before, after)
                val_loss += criterion(outputs, masks).item()
                val_iou += calculate_iou(outputs, masks)

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)

        print(f"\n>> Epoch [{epoch}/2] Summary | Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}", flush=True)

        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), "best_siamese_model.pth")
            print(f"  --> Saved checkpoint to 'best_siamese_model.pth' (Val IoU: {best_val_iou:.4f})\n", flush=True)

    print("\nFast training complete! Weights saved successfully.", flush=True)

if __name__ == "__main__":
    fast_train()

