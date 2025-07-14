import os
import shutil
from pathlib import Path

# Paths
input_root = "./"
output_root = "../PSR_final"  # We will write into the same dataset directory
splits = ["train", "val"]  # You can add 'test' if needed
skip = ["Simple"]

# Ensure target folders exist
for split in splits:
    os.makedirs(os.path.join(output_root, split), exist_ok=True)

# Go through each source
for source in os.listdir(input_root):
    source_path = os.path.join(input_root, source)
    if not os.path.isdir(source_path):
        continue

    if source in skip:
        continue

    for split in splits:
        split_path = os.path.join(source_path, split)
        if not os.path.isdir(split_path):
            continue

        for sample in os.listdir(split_path):
            src_sample_path = os.path.join(split_path, sample)

            # Create a unique sample name to avoid collision
            new_sample_name = f"{source}_{sample}"
            dst_sample_path = os.path.join(output_root, split, new_sample_name)

            # Copy the sample folder
            if os.path.exists(dst_sample_path):
                print(f"⚠️ Skipped duplicate: {dst_sample_path}")
            else:
                shutil.copytree(src_sample_path, dst_sample_path)
                print(f"✅ Copied: {src_sample_path} → {dst_sample_path}")
