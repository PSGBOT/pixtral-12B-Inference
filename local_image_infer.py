import base64
import requests
import os
from mistralai import Mistral

# Path to your images
origin_image_path = "/home/cyl/Reconst/Semantic-SAM/input/Pasted image (3).png"
mask_image_path = "/home/cyl/Reconst/Semantic-SAM/output/part_seg_dataset/id 4/mask_1.png"

# Getting the base64 strings
base64_original = encode_image(origin_image_path)
base64_mask = encode_image(mask_image_path)

if not base64_original or not base64_mask:
    print("Error: Failed to encode one or both images.")
    exit(1)

# Retrieve the API key from environment variables
api_key = os.environ.get("MISTRAL_API_KEY")
if not api_key:
    print("Error: MISTRAL_API_KEY environment variable not set.")
    exit(1)

# Specify model
model = "pixtral-12b-2409"

# Initialize the Mistral client
client = Mistral(api_key=api_key)

# Define the messages for the chat
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "I'm showing you two images. The first is an original image, and the second is a mask highlighting a specific object in the image. Please provide a detailed description of the masked object - what it is, its characteristics, and its context in the original image."
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_original}"
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{base64_mask}"
            }
        ]
    }
]

# Get the chat response
chat_response = client.chat.complete(
    model=model,
    messages=messages
)

# Print the content of the response
print("\nDescription of the masked object:")
print("-" * 50)
print(chat_response.choices[0].message.content)
print("-" * 50)
