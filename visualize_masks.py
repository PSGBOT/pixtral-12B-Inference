import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import argparse
import random


def get_beautiful_colors(n_colors):
    """Generate a beautiful color palette suitable for research papers"""
    # Professional color palette inspired by scientific visualization
    base_colors = [
        "#E74C3C",  # Red
        "#3498DB",  # Blue
        "#2ECC71",  # Green
        "#F39C12",  # Orange
        "#9B59B6",  # Purple
        "#1ABC9C",  # Turquoise
        "#E67E22",  # Carrot
        "#34495E",  # Dark blue-gray
        "#F1C40F",  # Yellow
        "#E91E63",  # Pink
        "#00BCD4",  # Cyan
        "#FF5722",  # Deep orange
    ]

    # Convert hex to RGB and normalize
    colors = []
    for i in range(n_colors):
        hex_color = base_colors[i % len(base_colors)]
        rgb = mcolors.hex2color(hex_color)
        colors.append(rgb)

    return colors


def get_random_color():
    """Get a single random color from the beautiful palette"""
    colors = get_beautiful_colors(12)  # Get all base colors
    return random.choice(colors)


def load_masks(mask_folder):
    """Load all mask images from folder that start with 'mask' or single mask file"""
    mask_path = Path(mask_folder)

    # Check if it's a single file
    if mask_path.is_file():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            return [mask], [mask_path.name]
        else:
            return [], []

    # If it's a directory, proceed with original logic
    mask_files = []

    # Support common image formats
    extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]

    for ext in extensions:
        # Only get files that start with "mask"
        mask_files.extend(mask_path.glob(f"mask*{ext}"))
        mask_files.extend(mask_path.glob(f"mask*{ext.upper()}"))

    masks = []
    filenames = []

    for mask_file in sorted(mask_files):
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            masks.append(mask)
            filenames.append(mask_file.name)

    return masks, filenames


def create_overlay_visualization(
    masks, filenames, colors, alpha=0.6, transparent_bg=False
):
    """Create an overlay visualization of all masks"""
    if not masks:
        raise ValueError("No masks found")

    # Get dimensions from first mask
    h, w = masks[0].shape

    # Create canvas (RGB or RGBA based on transparent_bg flag)
    if transparent_bg:
        canvas = np.zeros((h, w, 4), dtype=np.float32)  # RGBA
    else:
        canvas = np.zeros((h, w, 3), dtype=np.float32)  # RGB

    # Overlay each mask with its color
    for i, (mask, color) in enumerate(zip(masks, colors)):
        # Normalize mask to 0-1
        mask_norm = mask.astype(np.float32) / 255.0

        if transparent_bg:
            # Create RGBA colored mask
            colored_mask = np.zeros((h, w, 4), dtype=np.float32)
            for c in range(3):
                colored_mask[:, :, c] = mask_norm * color[c]
            colored_mask[:, :, 3] = mask_norm * alpha  # Alpha channel

            # For transparent background, add masks directly without blending background
            mask_weight = mask_norm[:, :, np.newaxis]
            canvas[:, :, :3] = (
                canvas[:, :, :3] * (1 - mask_weight)
                + colored_mask[:, :, :3] * mask_weight
            )
            canvas[:, :, 3] = np.maximum(canvas[:, :, 3], colored_mask[:, :, 3])
        else:
            # Create RGB colored mask
            colored_mask = np.zeros((h, w, 3), dtype=np.float32)
            for c in range(3):
                colored_mask[:, :, c] = mask_norm * color[c]

            # Add to canvas with alpha blending
            mask_weight = mask_norm[:, :, np.newaxis] * alpha
            canvas = canvas * (1 - mask_weight) + colored_mask * mask_weight

    return np.clip(canvas, 0, 1)


def create_grid_visualization(masks, filenames, colors, ncols=3):
    """Create a grid visualization showing individual masks"""
    n_masks = len(masks)
    nrows = (n_masks + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_masks):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]

        # Create colored mask
        mask = masks[i].astype(np.float32) / 255.0
        colored_mask = np.zeros((*mask.shape, 3))
        for c in range(3):
            colored_mask[:, :, c] = mask * colors[i][c]

        ax.imshow(colored_mask)
        ax.set_title(f"{filenames[i]}", fontsize=10, pad=10)
        ax.axis("off")

    # Hide empty subplots
    for i in range(n_masks, nrows * ncols):
        row, col = i // ncols, i % ncols
        axes[row, col].axis("off")

    plt.tight_layout()
    return fig


def visualize_masks(
    mask_folder,
    output_path=None,
    visualization_type="overlay",
    alpha=0.6,
    dpi=300,
    transparent_bg=False,
):
    """
    Main function to visualize masks

    Args:
        mask_folder: Path to folder containing masks
        output_path: Path to save visualization (optional)
        visualization_type: 'overlay' or 'grid'
        alpha: Transparency for overlay mode
        dpi: Resolution for saved image
        transparent_bg: Whether to use transparent background (overlay mode only)
    """
    # Load masks
    masks, filenames = load_masks(mask_folder)

    if not masks:
        print("No mask images found in the specified folder")
        return

    print(f"Found {len(masks)} masks: {filenames}")

    # Generate colors
    if len(masks) == 1:
        # If single mask, use random color
        colors = [get_random_color()]
    else:
        # If multiple masks, use sequential colors
        colors = get_beautiful_colors(len(masks))

    if visualization_type == "overlay":
        # Create overlay visualization
        result = create_overlay_visualization(
            masks, filenames, colors, alpha, transparent_bg
        )

        # Plot
        plt.figure(figsize=(12, 8))
        plt.imshow(result)
        plt.axis("off")

        # Set transparent background for the figure if transparent_bg is enabled
        if transparent_bg:
            plt.gcf().patch.set_alpha(0.0)
            plt.gca().patch.set_alpha(0.0)

        # # Add legend
        # legend_elements = [
        #     plt.Rectangle(
        #         (0, 0), 1, 1, facecolor=colors[i], alpha=0.8, label=filenames[i]
        #     )
        #     for i in range(len(masks))
        # ]
        # plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.15, 1))

    elif visualization_type == "grid":
        # Create grid visualization
        fig = create_grid_visualization(masks, filenames, colors)
        plt.suptitle("Individual Mask Visualization", fontsize=16)

    # Save if output path provided
    if output_path:
        if transparent_bg and visualization_type == "overlay":
            plt.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="none",
                edgecolor="none",
                transparent=True,
            )
        else:
            plt.savefig(
                output_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
            )
        print(f"Visualization saved to {output_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize masks with beautiful colors"
    )
    parser.add_argument("mask_folder", help="Path to folder containing mask images")
    parser.add_argument("--output", "-o", help="Output path for saved visualization")
    parser.add_argument(
        "--type",
        "-t",
        choices=["overlay", "grid"],
        default="overlay",
        help="Visualization type (default: overlay)",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=1,
        help="Transparency for overlay mode (default: 0.6)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="DPI for saved image (default: 300)"
    )
    parser.add_argument(
        "--transparent",
        action="store_true",
        help="Use transparent background (overlay mode only)",
    )

    args = parser.parse_args()

    visualize_masks(
        mask_folder=args.mask_folder,
        output_path=args.output,
        visualization_type=args.type,
        alpha=args.alpha,
        dpi=args.dpi,
        transparent_bg=args.transparent,
    )


if __name__ == "__main__":
    main()
