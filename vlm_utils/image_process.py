import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import time  # Import time module for performance measurement
from PIL import Image, ImageDraw, ImageFont
import os
import json


class crop_config:
    def __init__(
        self, crop=False, bbox=[641, 343, 754, 453], padding_box=[-10, -10, 10, 10]
    ):
        self.crop = crop
        self.bbox = bbox
        self.padding_box = padding_box


def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: The file {image_path} was not found.")
        return None
    except Exception as e:  # Added general exception handling
        print(f"Error: {e}")
        return None


def addContour(image, mask):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image


def dim_and_highlight(image, mask, dim_level=0.7, highlight_level=0.3):
    """
    Create a dimming effect on non-instance areas and highlight instance areas with a green tint.

    Args:
        image (PIL.Image): RGBA image to process
        mask (numpy.ndarray): 2D boolean array where True represents instance pixels
        dim_level (float): Opacity level for dimming non-instance areas (0-1)
        highlight_level (float): Intensity of green highlight on instance areas (0-1)

    Returns:
        PIL.Image: Image with dimmed non-instance areas and highlighted instance areas
    """
    # Convert PIL image to numpy array for processing
    image_array = np.array(image)

    # Step 1: Create dimming effect for non-instance areas
    # Convert mask to a format usable for creating an alpha mask
    alpha_mask = np.zeros(image.size[::-1], dtype=np.uint8)
    alpha_mask[~mask] = dim_level * 255  # Set alpha value for non-instance areas

    # Create a dimming layer
    dim_array = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    dim_array[..., 3] = alpha_mask  # Set alpha channel

    # Convert numpy array to PIL Image
    dim_layer = Image.fromarray(dim_array, mode="RGBA")

    # Composite the dim layer onto the image
    dimmed_image = Image.alpha_composite(image, dim_layer)

    # Step 2: Create green highlight effect for instance areas
    # Create a green highlight layer
    highlight_array = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    # Set green channel for instance areas
    highlight_array[mask, 1] = int(255 * highlight_level)  # Green channel
    highlight_array[mask, 3] = int(
        255 * highlight_level
    )  # Alpha channel for transparency

    # Convert numpy array to PIL Image
    highlight_layer = Image.fromarray(highlight_array, mode="RGBA")

    # Composite the highlight layer onto the dimmed image
    final_image = Image.alpha_composite(dimmed_image, highlight_layer)

    # Convert the processed image to RGB for saving
    return final_image.convert("RGB")


def process_image_for_description(
    image_path,
    mask_path,
    mask_level=0.7,
    highlight_level=0.3,
    crop_config=crop_config(),
    debug=True,
):
    """
    Process an image with its mask.

    Args:
        image_path (str): Path to the image file
        mask_path (str): Path to the mask file

    Returns:
        str: Base64 encoded processed image or None if error occurs
    """
    try:
        # Start timing the process
        start_time = time.time()

        # Read the image and mask
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Get mask data as numpy array for processing
        if mask.size != image.size:
            image = image.resize(mask.size, Image.LANCZOS)

        mask_data = np.array(mask)
        instance_mask = mask_data > 128

        # Process the image
        processed_image = image.copy()
        processed_image = dim_and_highlight(
            processed_image, instance_mask, mask_level, highlight_level
        )
        processed_np = np.array(processed_image)

        # Add contours
        contoured_image = addContour(processed_np, mask_data)

        # Convert back to PIL
        processed_image = Image.fromarray(contoured_image)

        # crop_box = tuple(a + b for a, b in zip(crop_box, padding_size))
        if crop_config.crop is True:
            bbox = crop_config.bbox
            padding_box = crop_config.padding_box
            bbox = tuple(a + b for a, b in zip(bbox, padding_box))
            processed_image = processed_image.crop(bbox)
        # Display the highlighted image

        if debug is True:
            processed_image.show()

        # Save the processed image to a BytesIO object
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the processed image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Calculate and print the total processing time
        end_time = time.time()
        if debug is True:
            print(f"Total image processing time: {end_time - start_time:.4f} seconds")

        return encoded_image
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def combined_image_present(vis_img, structured_kinematic_desc):
    mask1 = vis_img[0]
    mask2 = vis_img[1]
    split_width = 4
    total_w = mask1.width + split_width + mask2.width
    max_h = max(mask1.height, mask2.height)
    combined = Image.new("RGBA", (total_w, max_h), "WHITE")
    combined.paste(mask1, (0, 0))
    draw = ImageDraw.Draw(combined)
    draw.rectangle(
        [mask1.width, 0, mask1.width + split_width, max_h],
        fill="white"
    )
    combined.paste(mask2, (mask1.width + split_width, 0))
    font = ImageFont.load_default()
    pad = 8

    lines = [f"{k}: {v}" for k, v in structured_kinematic_desc.items()]

    line_h = 2*int(font.getlength("A"))
    text_h = line_h * len(lines)

    total_h = max_h + text_h + 2*pad
    old = combined
    combined = Image.new("RGBA", (total_w, total_h), "WHITE")

    combined.paste(old, (0, 0))

    draw = ImageDraw.Draw(combined)
    y = max_h + pad
    for line in lines:
        draw.text((pad, y), line, fill="black", font=font)
        y += line_h

    combined.show()