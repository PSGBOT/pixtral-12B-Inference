import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral
import json

class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir):
        self.dataset_dir = dataset_dir
        self.dataset = {}
    def load_dataset(self):
        try:
            if not os.path.exists(self.dataset_dir):
                print(f"Dataset file not found: {self.dataset_dir}")
                return None

            with open(self.dataset_dir, 'r') as f:
                dataset = json.load(f)

            print(f"Loaded part segmentation dataset from {self.dataset_dir}")
            print(f"Dataset contains {len(dataset)} images")

            self.level_seg_dataset = dataset
            return self.level_seg_dataset

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None

    def load_children_pairs(self, image_id, ):





if __name__ == "__main__":
    image_path = "/home/cyl/Reconst/Semantic-SAM/input/id 2.png" # original image
    mask_path = "/home/cyl/Reconst/Semantic-SAM/output/part_seg_dataset/id 2/mask0/mask_0.png" # mask
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
