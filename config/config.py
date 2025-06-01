"""
Configuration file for Pixtral-12B-Inference project.
Contains paths and settings used across the project.
"""

# Model settings
VLM_SETTINGS = {
    "model_name": "pixtral-12b-2409",
    "max_tokens": 4096,
    "temperature": 0.5,
}

LLM_SETTINGS = {
    "model_name": "open-mistral-nemo-2407",
    "max_tokens": 4096,
    "temperature": 0.3,
}

# Output settings
OUTPUT_SETTINGS = {
    "save_processed_images": True,
    "output_dir": "output",
}
