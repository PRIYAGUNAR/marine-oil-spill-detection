"""
Prediction script for U-Net Oil Spill Detection model.

Loads a trained model and visualizes predictions on sample images.
Saves a grid showing: [Original SAR Image] | [Ground Truth Mask] | [Model Prediction]
"""

import os
import argparse
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import OilSpillDataset
from model import UNet

@torch.no_grad()
def predict_and_visualize(model, dataset, device, num_samples=5, output_dir="predictions", threshold=0.5):
    """Run model on random samples and save visualization grid."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Get random indices
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = [axes]
        
    for i, idx in enumerate(tqdm(indices, desc="Generating predictions")):
        # Get raw data
        image, mask = dataset[idx]
        image_name = dataset.filenames[idx]
        
        # Add batch dim, move to device
        image_batch = image.unsqueeze(0).to(device)
        
        # Forward pass
        logits = model(image_batch)
        prob = torch.sigmoid(logits)
        pred_bin = (prob > threshold).float()
        
        # Move back to CPU and remove batch/channel dims
        img_np = image.squeeze(0).cpu().numpy()
        mask_np = mask.squeeze(0).cpu().numpy()
        pred_np = pred_bin.squeeze().cpu().numpy()
        prob_np = prob.squeeze().cpu().numpy()
        
        # Overlay oil spill probability on image for better context
        overlay = np.dstack([img_np, img_np, img_np]) # RGB grayscale
        # Add red tint where model is highly confident (prob > 0.5)
        overlay[..., 0] = np.clip(overlay[..., 0] + prob_np * 0.5, 0, 1) # Red channel
        
        # Plot Image
        axes[i][0].imshow(img_np, cmap="gray")
        axes[i][0].set_title(f"SAR Input: {image_name}")
        axes[i][0].axis("off")
        
        # Plot Ground Truth
        axes[i][1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
        axes[i][1].set_title("Ground Truth Mask\n(White=Oil, Black=Water)")
        axes[i][1].axis("off")
        
        # Plot Prediction Overlay
        axes[i][2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
        axes[i][2].set_title("Model Prediction\n(Threshold > 0.5)")
        axes[i][2].axis("off")

    plt.tight_layout()
    save_path = os.path.join(output_dir, "prediction_samples.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n ✓ Saved {num_samples} sample predictions to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize U-Net predictions")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(os.path.dirname(__file__),
                                             "split_dataset-20260310T093208Z-1-001",
                                             "split_dataset"),
                        help="Path to split_dataset/ folder")
    parser.add_argument("--model_path", type=str, default="best_model.pth",
                        help="Path to saved model weights")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to visualize")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of random samples to visualize")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Directory to save the output image")
    args = parser.parse_args()

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f" U-Net Visual Predictions")
    print(f"{'='*60}")
    print(f" Device:     {device}")
    print(f" Model:      {args.model_path}")
    
    if not os.path.exists(args.model_path):
        print(f"\n[ERROR] Model file not found: {args.model_path}")
        print("Please train the model first before generating predictions.")
        return

    # ── Dataset ──
    dataset = OilSpillDataset(args.data_dir, split=args.split)

    # ── Load Model ──
    model = UNet(in_channels=1, out_channels=1).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # ── Visualize ──
    predict_and_visualize(model, dataset, device, num_samples=args.samples, output_dir=args.out_dir)
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
