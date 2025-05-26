import message
import os
from mistralai import Mistral

if __name__ == "__main__":
    image_path = "/home/cyl/Reconst/Semantic-SAM/input/Pasted image (3).png"
    mask_path = "/home/cyl/Reconst/Semantic-SAM/output/part_seg_dataset/id 4/mask_1.png"
    msg = message.instance_description_msg_pil(image_path, mask_path)

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
    print("\nDescription of the masked object:")
    print("-" * 50)
    print(chat_response.choices[0].message.content)
    print("-" * 50)
