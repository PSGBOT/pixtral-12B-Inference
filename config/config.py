"""
Configuration file for Pixtral-12B-Inference project.
Contains paths and settings used across the project.
"""

# Model settings
FLASH_VLM_SETTINGS = {
    "model_name": "gemini-2.0-flash",
    "max_tokens": 4096,
    "temperature": 0.2,
}
SOTA_VLM_SETTINGS = {
    "model_name": "gemini-2.5-flash-preview-05-20",
    "max_tokens": 4096,
    "temperature": 0.7,
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

VLM_SETTINGS_MIS = {
    "model_name": "pixtral-12b-2409",
    "max_tokens": 4096,
    "temperature": 0.5,
}

LLM_SETTINGS_MIS = {
    "model_name": "open-mistral-nemo-2407",
    "max_tokens": 4096,
    "temperature": 0.3,
}
