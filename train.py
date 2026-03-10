"""
Training script for U-Net Oil Spill Detection model.

Usage:
    python train.py                          # Full training (30 epochs)
    python train.py --epochs 1 --batch_size 2  # Quick test run
"""

import os
import json
import csv
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OilSpillDataset
from model import UNet


# ─────────────────────────────────────────────────────────────────────────────
# IoU Metric
# ─────────────────────────────────────────────────────────────────────────────

def compute_iou(preds, targets, threshold=0.5, smooth=1e-6):
    """
    Compute Intersection over Union for the oil spill (positive) class.
    
    Args:
        preds: Raw logits from the model (B, 1, H, W).
        targets: Ground truth binary masks (B, 1, H, W).
        threshold: Threshold to binarize predictions.
        smooth: Smoothing factor to avoid division by zero.
    
    Returns:
        IoU score (float).
    """
    preds_bin = (torch.sigmoid(preds) > threshold).float()
    
    intersection = (preds_bin * targets).sum()
    union = preds_bin.sum() + targets.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ─────────────────────────────────────────────────────────────────────────────
# Training & Validation loops
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Run one training epoch. Returns average loss and IoU."""
    model.train()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_iou += compute_iou(outputs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_iou / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Run validation. Returns average loss and IoU."""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        running_loss += loss.item() * images.size(0)
        running_iou += compute_iou(outputs, masks) * images.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_iou / n


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train U-Net for Oil Spill Detection")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "split_dataset-20260310T093208Z-1-001",
                                             "split_dataset"),
                        help="Path to split_dataset/ folder")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process, recommended on Windows)")
    parser.add_argument("--save_path", type=str, default="best_model.pth",
                        help="Path to save best model")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" U-Net Oil Spill Detection — Training")
    print(f"{'='*60}")
    print(f" Device:     {device}")
    print(f" Epochs:     {args.epochs}")
    print(f" Batch Size: {args.batch_size}")
    print(f" LR:         {args.lr}")
    print(f" Patience:   {args.patience}")

    # ── Datasets & DataLoaders ──
    train_ds = OilSpillDataset(args.data_dir, split="train")
    val_ds = OilSpillDataset(args.data_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # ── Model ──
    model = UNet(in_channels=1, out_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Parameters: {total_params:,}")

    # ── Loss with class weight ──
    weights_path = os.path.join(args.data_dir, "class_weights.json")
    if os.path.exists(weights_path):
        with open(weights_path, "r") as f:
            weights = json.load(f)
        pos_weight_val = weights["oil_spill"]["weight"]
    else:
        pos_weight_val = 49.74  # fallback
        print(f" [WARNING] class_weights.json not found, using default pos_weight={pos_weight_val}")

    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f" Pos Weight: {pos_weight_val}")

    # ── Optimizer & Scheduler ──
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    print(f"{'='*60}\n")

    # ── Training Log CSV ──
    log_path = os.path.join(os.path.dirname(args.save_path) or ".", "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_iou", "val_loss", "val_iou", "lr"])

    # ── Training Loop ──
    best_val_iou = 0.0
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # Scheduler step
        scheduler.step(val_iou)

        elapsed = time.time() - epoch_start

        # Print epoch summary
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  IoU: {train_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}  IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.0f}s")

        # Log to CSV
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{train_iou:.6f}",
                             f"{val_loss:.6f}", f"{val_iou:.6f}", f"{current_lr:.2e}"])

        # Save best model
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_loss": val_loss,
            }, args.save_path)
            print(f"  ✓ Best model saved (Val IoU: {val_iou:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= args.patience:
            print(f"\n  ⚠ Early stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"\n{'='*60}")
    print(f" Training complete!")
    print(f" Best Val IoU: {best_val_iou:.4f}")
    print(f" Model saved:  {os.path.abspath(args.save_path)}")
    print(f" Log saved:    {os.path.abspath(log_path)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
