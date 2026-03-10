"""
Dataset loader for Marine Oil Spill Detection.
Loads preprocessed SAR images and binary masks, converts to PyTorch tensors.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    PyTorch Dataset for oil spill segmentation.
    
    Images are loaded as grayscale (SAR data is single-channel),
    normalized to [0, 1]. Masks are binarized to {0, 1}.
    """

    def __init__(self, root_dir, split="train"):
        """
        Args:
            root_dir: Path to split_dataset/ folder containing train/val/test.
            split: One of 'train', 'val', 'test'.
        """
        self.image_dir = os.path.join(root_dir, split, "images")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        # Get sorted list of filenames (images and masks share the same names)
        self.filenames = sorted([
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

        print(f"[{split.upper()}] Loaded {len(self.filenames)} image-mask pairs")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]

        # --- Load image as grayscale ---
        img_path = os.path.join(self.image_dir, fname)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # --- Load mask as grayscale ---
        mask_path = os.path.join(self.mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")

        # --- Normalize image to [0, 1] ---
        image = image.astype(np.float32) / 255.0

        # --- Binarize mask: any pixel > 127 becomes 1 (oil spill), else 0 (clean water) ---
        mask = (mask > 127).astype(np.float32)

        # --- Convert to tensors with channel dimension: (1, H, W) ---
        image = torch.from_numpy(image).unsqueeze(0)  # (1, 512, 512)
        mask = torch.from_numpy(mask).unsqueeze(0)     # (1, 512, 512)

        return image, mask


if __name__ == "__main__":
    # Quick sanity check
    dataset_root = os.path.join(
        os.path.dirname(__file__),
        "split_dataset-20260310T093208Z-1-001", "split_dataset"
    )

    for split in ["train", "val", "test"]:
        ds = OilSpillDataset(dataset_root, split)
        img, msk = ds[0]
        print(f"  Image shape: {img.shape}, dtype: {img.dtype}, "
              f"range: [{img.min():.3f}, {img.max():.3f}]")
        print(f"  Mask shape:  {msk.shape}, dtype: {msk.dtype}, "
              f"unique values: {torch.unique(msk).tolist()}")
