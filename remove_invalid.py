import cv2
import json
import os
import networkx as nx
import argparse
import numpy as np
import shutil  # Import shutil for moving directories
from PIL import Image


def maskname_to_index(maskname="mask0"):
    return int(maskname.split("mask")[1])


def _process_image_and_masks(sample_path):
    # center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    # scale = max(height, width) * 1.0
    # Apply flipping to src_img

    # Initialize 3 RGB channels for masks
    combined_masks_channels = np.zeros((512, 512, 3), dtype=np.float32)

    # Load and process individual masks
    masks_dir = sample_path
    masks_bbox = {}

    # Count total masks first to generate appropriate colors
    mask_files = [f for f in os.listdir(masks_dir) if f.startswith("mask")]
    num_masks = len(mask_files)

    if os.path.exists(masks_dir):
        # Store individual masks for overlap resolution
        individual_masks = {}

        # First pass: load all masks without combining
        for mask_filename in mask_files:
            mask_path = os.path.join(masks_dir, mask_filename)
            mask = Image.open(mask_path).convert("L")  # Load as grayscale
            # Resize mask to original image size before any transformations
            mask = mask.resize((512, 512), Image.BILINEAR)
            mask = np.asarray(mask)

            # Now, process the mask to set border to 128 and inside to 255
            # Ensure mask is binary (0 or 255) for findContours
            mask_binary = (mask > 0).astype(np.uint8) * 255

            mask_index = maskname_to_index(os.path.splitext(mask_filename)[0])

            # Store the processed mask for overlap resolution
            individual_masks[mask_index] = (mask_binary > 0).astype(np.uint8)

        # Second pass: resolve overlaps using boolean subtraction
        # Lower index masks have priority over higher index masks
        resolved_masks = {}
        sorted_mask_indices = sorted(individual_masks.keys())

        for i, mask_idx in enumerate(sorted_mask_indices):
            current_mask = individual_masks[mask_idx].copy()

            # Subtract all previously processed masks (lower indices have priority)
            for j in range(i):
                prev_mask_idx = sorted_mask_indices[j]
                if prev_mask_idx in resolved_masks:
                    # Boolean subtraction: current_mask = current_mask AND NOT prev_mask
                    current_mask = cv2.bitwise_and(
                        current_mask, cv2.bitwise_not(resolved_masks[prev_mask_idx])
                    )

            resolved_masks[mask_idx] = current_mask

            # Update bbox information based on resolved mask
            if np.any(current_mask):
                return True
            else:
                # If mask is completely removed by subtraction, set empty bbox
                print("invalid mask")
                return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Remove the training sample with mask for background"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the PSR dataset"
    )
    parser.add_argument(
        "--trash_dir",
        type=str,
        required=True,
        help="Path to the removed PSR dataset sample",
    )

    args = parser.parse_args()

    trash_dataset_dir = args.trash_dir
    if not os.path.exists(trash_dataset_dir):
        print(f"Creating trash dataset directory: {trash_dataset_dir}")
        os.makedirs(trash_dataset_dir)

    sample_list = os.listdir(args.dataset_dir)
    for sample_name in sample_list:
        print(f"Processing {sample_name}...")
        sample_dir = os.path.join(args.dataset_dir, sample_name)
        config_path = os.path.join(sample_dir, "config.json")

        remove_sample = not _process_image_and_masks(sample_dir)
        if remove_sample:
            print(f"Sample {sample_name} should be removed.")
            destination_path = os.path.join(trash_dataset_dir, sample_name)
            try:
                shutil.move(sample_dir, destination_path)
                print(f"Moved {sample_name} to {destination_path}")
            except Exception as e:
                print(f"Error moving {sample_name}: {e}")
        else:
            print(f"Sample {sample_name} should be kept.")
            # Optionally, you could move "kept" samples to a different "kept" directory
            # or just leave them in the original dataset_dir.
            # For now, we'll leave them in the original directory if not removed.

        # The print(masks) line is not very useful here, can be removed or commented out.
        # print(masks)
