"""
Configuration file for Pixtral-12B-Inference project.
Contains paths and settings used across the project.
"""

# Model settings
FLASH_VLM_SETTINGS = {
    "model_name": "gemini-2.0-flash",
    "max_tokens": 4096,
    "temperature": 1,
}
SOTA_VLM_SETTINGS = {
    "model_name": "gemini-2.0-flash",
    "max_tokens": 4096,
    "temperature": 1,
}

LLM_SETTINGS = {
    "model_name": "gemini-2.0-flash",
    "max_tokens": 4096,
    "temperature": 0.3,
}

# Output settings
OUTPUT_SETTINGS = {
    "save_processed_images": True,
    "output_dir": "output",
}
