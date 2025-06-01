import pixtral_utils.message as vlm_message
import os
import argparse
from mistralai import Mistral
from pixtral_utils.output_structure import Instance, Part
import sys

# Import config settings
from config import IMAGE_PATHS, MODEL_SETTINGS, OUTPUT_SETTINGS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate descriptions using Pixtral-12B model"
    )
    parser.add_argument("--image", help="Path to the image file (overrides config)")
    parser.add_argument("--mask", help="Path to the mask file (overrides config)")
    parser.add_argument(
        "--object_name", help="Name of the object for part description", default=""
    )
    parser.add_argument(
        "--mode",
        choices=["part", "instance"],
        default="instance",
        help="Description mode: 'part' or 'instance'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()

    # Get paths from command line args or config file
    image_path = args.image if args.image else IMAGE_PATHS["sample_image"]
    mask_path = args.mask if args.mask else IMAGE_PATHS["sample_mask"]

    # Fix path separators for cross-platform compatibility
    image_path = os.path.normpath(image_path)
    mask_path = os.path.normpath(mask_path)

    # Create appropriate message based on mode
    if args.mode == "part":
        if not args.object_name:
            print("Error: --object_name is required for part description mode.")
            exit(1)
        msg = vlm_message.part_description_msg(image_path, mask_path, args.object_name)
        response_format = Part
        print(f"Generating description for part: {args.object_name}")
    else:
        msg = vlm_message.instance_description_msg(image_path, mask_path)
        response_format = Instance
        print("Generating instance description")

    # Get API key from environment
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        exit(1)

    # Get model settings from config
    model = MODEL_SETTINGS["model_name"]
    max_tokens = MODEL_SETTINGS["max_tokens"]
    temperature = MODEL_SETTINGS["temperature"]

    print(f"Using model: {model}")
    print(f"Processing image: {image_path}")
    print(f"Using mask: {mask_path}")

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)

    # Generate the description
    chat_response = client.chat.parse(
        model=model,
        messages=[msg],
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Print the content of the response
    print("\nDescription of the masked part:")
    print("-" * 50)
    print(chat_response.choices[0].message.content)
    print("-" * 50)
