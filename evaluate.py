"""
Evaluation script for U-Net Oil Spill Detection model.

Evaluates a trained model on the test set using multiple metrics:
- Intersection over Union (IoU)
- Dice Coefficient (F1 Score)
- Precision
- Recall
- Pixel Accuracy
"""

import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OilSpillDataset
from model import UNet

def compute_metrics(preds_bin, targets, smooth=1e-6):
    """
    Compute binary classification metrics for a batch.
    Targets and preds_bin should be `{0, 1}`.
    """
    # Flatten tensors
    preds_flat = preds_bin.view(-1)
    targets_flat = targets.view(-1)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1 - targets_flat)).sum().item()
    fn = ((1 - preds_flat) * targets_flat).sum().item()
    tn = ((1 - preds_flat) * (1 - targets_flat)).sum().item()

    # Metrics
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth) # Same as F1
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    }

@torch.no_grad()
def evaluate_model(model, loader, device, threshold=0.5):
    """Run full evaluation on a DataLoader."""
    model.eval()
    
    total_metrics = {
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
        "tp": 0, "fp": 0, "fn": 0, "tn": 0
    }
    
    num_batches = len(loader)
    
    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds_bin = (probs > threshold).float()
        
        # Compute metrics for this batch
        batch_metrics = compute_metrics(preds_bin, masks)
        
        # Accumulate
        for k in total_metrics:
            total_metrics[k] += batch_metrics[k]
            
    # Average across batches for ratios
    for k in ["iou", "dice", "precision", "recall", "accuracy"]:
        total_metrics[k] /= num_batches
        
    return total_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained U-Net model")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "split_dataset-20260310T093208Z-1-001",
                                             "split_dataset"),
                        help="Path to split_dataset/ folder")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Path to saved model weights")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for probabilities")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" U-Net Model Evaluation")
    print(f"{'='*60}")
    print(f" Device:     {device}")
    print(f" Model:      {args.model_path}")
    print(f" Split:      {args.split.upper()}")
    print(f" Threshold:  {args.threshold}")

    if not os.path.exists(args.model_path):
        print(f"\n[ERROR] Model file not found: {args.model_path}")
        print("Please train the model first or provide the correct path.")
        return

    # ── Dataset & DataLoader ──
    dataset = OilSpillDataset(args.data_dir, split=args.split)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ── Load Model ──
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f" ✓ Model loaded (trained for {checkpoint.get('epoch', 'N/A')} epochs, "
          f"val IoU: {checkpoint.get('val_iou', 0.0):.4f})")
    
    print(f"{'='*60}\n")
    
    # ── Evaluate ──
    metrics = evaluate_model(model, loader, device, threshold=args.threshold)
    
    # ── Print Results ──
    print(f"\n{'='*60}")
    print(f" Evaluation Results ({args.split.upper()} set)")
    print(f"{'='*60}")
    print(f" Metric        | Score")
    print(f" --------------+---------")
    print(f" IoU           | {metrics['iou']:.4f}")
    print(f" Dice (F1)     | {metrics['dice']:.4f}")
    print(f" Precision     | {metrics['precision']:.4f}")
    print(f" Recall        | {metrics['recall']:.4f}")
    print(f" Pixel Acc     | {metrics['accuracy']:.4f}")
    print(f" --------------+---------")
    
    # Global confusion matrix over all pixels
    total_pixels = metrics['tp'] + metrics['fp'] + metrics['fn'] + metrics['tn']
    print(f"\n Confusion Matrix (Total Pixels: {total_pixels:,}):")
    print(f"                  Predicted Oil  | Predicted Clean")
    print(f" Actual Oil   : {metrics['tp']:14,d} | {metrics['fn']:14,d}")
    print(f" Actual Clean : {metrics['fp']:14,d} | {metrics['tn']:14,d}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
