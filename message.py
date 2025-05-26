import base64
import requests
import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageEnhance
from io import BytesIO
from mistralai import Mistral

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

def apply_mask_torch(image_path, mask_path):
    """
    Apply a mask to an image using PyTorch for efficient processing.

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image (white areas indicate the instance)

    Returns:
        PIL.Image: Processed image with highlighted instance
    """
    try:
        # Read the image and mask
        image = Image.open(image_path).convert("RGBA")
        mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale

        # Resize mask to match image dimensions if needed
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.LANCZOS)

        # Convert to PyTorch tensors
        to_tensor = T.ToTensor()
        image_tensor = to_tensor(image)  # Shape: [4, H, W] for RGBA
        mask_tensor = to_tensor(mask)    # Shape: [1, H, W]

        # Threshold the mask (values > 0.5 are considered part of the instance)
        instance_mask = (mask_tensor > 0.5).float()

        # Create a dimming tensor (semi-transparent black)
        dim_tensor = torch.zeros_like(image_tensor)
        dim_tensor[3, :, :] = 0.8  # Alpha channel set to 0.8 (semi-transparent)

        # Apply the mask: keep instance areas from original image, dim the rest
        # Expand mask to match image channels
        expanded_mask = instance_mask.expand_as(image_tensor)

        # Combine: original image where mask is 1, dim layer where mask is 0
        result_tensor = image_tensor * expanded_mask + dim_tensor * (1 - expanded_mask)

        # Convert back to PIL Image
        to_pil = T.ToPILImage()
        processed_image = to_pil(result_tensor)

        # Convert to RGB for compatibility
        if processed_image.mode == "RGBA":
            background = Image.new("RGB", processed_image.size, (0, 0, 0))
            background.paste(processed_image, mask=processed_image.split()[3])
            processed_image = background

        return processed_image

    except Exception as e:
        print(f"Error in PyTorch mask application: {e}")
        return None

def instance_description_msg(image_path, mask_path):
    """
    Process an image with its mask to highlight an instance.

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image (white areas indicate the instance)

    Returns:
        dict: A message structure for the API with the processed image
    """
    try:
        # Use the PyTorch-based function to apply the mask
        processed_image = apply_mask_torch(image_path, mask_path)

        if processed_image is None:
            raise Exception("Failed to process image with PyTorch")

        # Display the highlighted image
        processed_image.show()

        # Save the processed image to a BytesIO object
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the processed image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create the message structure for the API
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this highlighted object in the image."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
            ]
        }
        return message

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        # Fall back to PIL-based processing if PyTorch method fails
        print("Falling back to PIL-based processing...")
        return instance_description_msg_pil(image_path, mask_path)

def instance_description_msg_pil(image_path, mask_path):
    """
    Process an image with its mask to highlight an instance using PIL.
    This is a fallback method if the PyTorch approach fails.

    Args:
        image_path (str): Path to the original image
        mask_path (str): Path to the mask image (white areas indicate the instance)

    Returns:
        dict: A message structure for the API with the processed image
    """
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
                    dim_draw.point((x, y), fill=(0, 0, 0, 230))

        # Composite the dim layer onto the image
        processed_image = Image.alpha_composite(processed_image, dim_layer)

        # Convert the processed image to RGB for saving
        processed_image = processed_image.convert("RGB")

        # Display the highlighted image
        processed_image.show()

        # Save the processed image to a BytesIO object
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the processed image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Create the message structure for the API
        message = {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this highlighted object in the image."},
                {"type": "image_url", "image_url": f"data:image/jpeg;base64,{encoded_image}"}
            ]
        }

        return message

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
