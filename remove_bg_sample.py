import cv2
import json
import os
import networkx as nx
import argparse
import numpy as np
import shutil  # Import shutil for moving directories

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
        masks = []
        for dir in os.listdir(sample_dir):
            if "mask" in dir:
                mask_img = cv2.imread(os.path.join(sample_dir, dir))
                masks.append(mask_img)

        single_thresh = 0.45
        union_thresh = 0.55

        remove_sample = False

        if len(masks) == 2:
            # get the union of masks
            mask_union = cv2.bitwise_or(masks[0], masks[1])
            # expand the mask union
            kernel = np.ones((5, 5), np.uint8)
            dilate_mask = cv2.dilate(mask_union, kernel, iterations=2)
            dilate_mask = cv2.cvtColor(dilate_mask, cv2.COLOR_BGR2GRAY)
            border_size = 1  # Define the border size in pixels
            height, width = dilate_mask.shape[:2]

            # Extract border regions
            top_border = dilate_mask[0:border_size, 0:width]
            bottom_border = dilate_mask[height - border_size : height, 0:width]
            left_border = dilate_mask[0:height, 0:border_size]
            right_border = dilate_mask[0:height, width - border_size : width]

            # Count non-zero pixels in border regions
            top_pixels = cv2.countNonZero(top_border)
            bottom_pixels = cv2.countNonZero(bottom_border)
            left_pixels = cv2.countNonZero(left_border)
            right_pixels = cv2.countNonZero(right_border)

            total_border_pixels = (
                top_pixels + bottom_pixels + left_pixels + right_pixels
            )
            print(f"Total non-zero border pixels: {total_border_pixels}")

            # Determine if the sample should be removed
            border_pixel_threshold = (2 * height + 2 * width) * union_thresh
            print(f"Border pixel threshold: {border_pixel_threshold}")

            if total_border_pixels > border_pixel_threshold:
                print(
                    f"Sample {sample_name} flagged for removal: High border pixel count."
                )
                remove_sample = True

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
