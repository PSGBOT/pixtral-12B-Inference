import base64
import numpy as np
from PIL import Image
from io import BytesIO
import cv2
import time  # Import time module for performance measurement

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

def dim(image, mask):
    """
    Create a dimming effect on non-instance areas of an image based on a mask.

    Args:
        image (PIL.Image): RGBA image to process
        mask (numpy.ndarray): 2D boolean array where True represents instance pixels

    Returns:
        PIL.Image: Image with dimmed non-instance areas
    """
    # Convert mask to a format usable for creating an alpha mask
    # Invert the mask since we want to dim areas where mask is False
    alpha_mask = np.zeros(image.size[::-1], dtype=np.uint8)
    alpha_mask[~mask] = 180  # Set alpha value for non-instance areas

    # Create a dimming layer
    dim_array = np.zeros((*image.size[::-1], 4), dtype=np.uint8)
    dim_array[..., 3] = alpha_mask  # Set alpha channel

    # Convert numpy array to PIL Image
    dim_layer = Image.fromarray(dim_array, mode="RGBA")

    # Composite the dim layer onto the image
    image = Image.alpha_composite(image, dim_layer)

    # Convert the processed image to RGB for saving
    return image.convert("RGB")


def process_image_for_description(image_path, mask_path, crop=None, debug=True):
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
        processed_image = dim(processed_image, instance_mask)
        processed_np = np.array(processed_image)

        # Add contours
        contoured_image = addContour(processed_np, mask_data)

        # Convert back to PIL
        processed_image = Image.fromarray(contoured_image)

        # Display the highlighted image
        #
        # if debug is True:
        #     processed_image.show()

        # Save the processed image to a BytesIO object
        buffer = BytesIO()
        processed_image.save(buffer, format="JPEG")
        buffer.seek(0)

        # Encode the processed image to base64
        encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Calculate and print the total processing time
        end_time = time.time()
        # if debug is True:
        #     print(f"Total image processing time: {end_time - start_time:.4f} seconds")

        return encoded_image
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        return None
    except Exception as e:
        print(f"Error processing image: {e}")
        return None
