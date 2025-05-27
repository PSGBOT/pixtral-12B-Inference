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

    def load_children_pairs(self, image_id):
