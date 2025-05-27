import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral

if __name__ == "__main__":
    image_path = "/home/cyl/Reconst/Semantic-SAM/input/id 2.png" # original image
    mask_path = "/home/cyl/Reconst/Semantic-SAM/output/part_seg_dataset/id 2/mask0/mask0.png" # mask
    # this message is for part description generation
    msg = vlm_message.part_description_msg(image_path, mask_path, "microwave oven")
    # this message is for instance description generation
    # msg = vlm_message.instance_description_msg(image_path, mask_path)

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        exit(1)

    # Specify model
    model = "pixtral-12b-2409"

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)
    chat_response = client.chat.complete(
        model=model,
        messages=[msg]
    )
    # Print the content of the response
    print("\nDescription of the masked part:")
    print("-" * 50)
    print(chat_response.choices[0].message.content)
    print("-" * 50)
