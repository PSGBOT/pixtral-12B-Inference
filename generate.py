import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral
from pydantic import BaseModel
class Instance(BaseModel):
    name: str
    usage: str

class Part(BaseModel):
    name: str
    purpose: str
    text: str

if __name__ == "__main__":

    image_path = "part_seg_dataset_sample\part_seg_dataset\id 3.png"
    mask_path = "part_seg_dataset_sample\part_seg_dataset\id 3\mask1\mask1\mask_1.png" # mask

    # this message is for part description generation
    msg = vlm_message.part_description_msg(image_path, mask_path, "soap dispenser pump")
    # this message is for instance description generation
    # msg = vlm_message.instance_description_msg(image_path, mask_path)

    api_key = "rMcpQanYgmqXsRMf2bvxnEcthE4PIdYd"
    if not api_key:
        print("Error: MISTRAL_API_KEY environment variable not set.")
        exit(1)

    # Specify model
    model = "pixtral-12b-2409"

    # Initialize the Mistral client
    client = Mistral(api_key=api_key)
    #chat_response = client.chat.complete(
    #    model=model,
    #    messages=[msg]
    #)
    chat_response = client.chat.parse(
        model=model,
        messages=[msg],
        response_format=Instance,
        max_tokens=256,
        temperature=0.5
    )
    # Print the content of the response
    print("\nDescription of the masked part:")
    print("-" * 50)
    print(chat_response.choices[0].message.content)
    print("-" * 50)
