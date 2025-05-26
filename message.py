import base64
import requests
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance
from io import BytesIO
from mistralai import Mistral
import cv2

def encode_image(image_path):
    """Encode the image to base64."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
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

def process_image(image_path, mask_path):
    try:
        # Read the image and mask
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Resize mask to match image dimensions if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        # Create a copy of the original image for processing
        processed_image = image.copy()

        # Create a dimming layer for dark areas of the mask
        dim_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        dim_draw = ImageDraw.Draw(dim_layer)

        # Get mask data as numpy array for processing
        mask_data = np.array(mask)

        # Find contours of the light areas (instance)
        # For simplicity, we'll consider pixels with value > 128 as the instance
        instance_mask = mask_data > 128

        # Create a dimming effect for non-instance areas
        for y in range(image.height):
            for x in range(image.width):
                if not instance_mask[y, x]:
                    # Add semi-transparent black pixel to dim non-instance areas
                    dim_draw.point((x, y), fill=(0, 0, 0, 130))

        # Composite the dim layer onto the image
        processed_image = Image.alpha_composite(processed_image, dim_layer)

        # Convert the processed image to RGB for saving
        processed_image = processed_image.convert("RGB")

        # Convert to numpy array for contour processing
        processed_np = np.array(processed_image)
        mask_np = np.array(mask)

        # Add contours
        contoured_image = addContour(processed_np, mask_np)

        # Convert back to PIL
        processed_image = Image.fromarray(contoured_image)

        # Display the highlighted image
        processed_image.show()

        # Save the processed image to a BytesIO object
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the processed image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return encoded_image
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None



def instance_description_msg(image_path, mask_path):
    """
    Process an image with its mask.
    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image (white areas indicate the instance)

    Returns:
        dict: A message structure for the API with the processed image
    """
    processed_image = process_image(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this highlighted object in the image"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }
    return message


def part_description_msg(image_path, mask_path, parent_description):
    processed_image = process_image(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Describe this highlighted part in the image, given that it is a part of a {parent_description}."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }

    return message
