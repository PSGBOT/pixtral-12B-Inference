import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral
import json
import argparse
from PIL import Image
import numpy as np

class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir):
        self.dataset_dir = dataset_dir
        self.src_image_dir = src_image_dir
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

    def load_children_pair_with_parent(self, p_dir, c_dir_1, c_dir_2):
        """
        load masks
        """
        pass
    def generate_relation(self, res_dir, child_pair, src_image):
        """
        Generate relation between two objects

        Args:
            res_dir: Directory to save results
            child_pair: Tuple of (child1_id, child2_id)
            src_image: Source image

        Returns:
            Dictionary containing relation information
        """
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        child1_id, child2_id = child_pair

        # Get object information from dataset
        image_id = os.path.basename(src_image.filename).split('.')[0]
        image_data = self.level_seg_dataset.get(image_id, {})

        objects = image_data.get("objects", {})
        child1_data = objects.get(child1_id, {})
        child2_data = objects.get(child2_id, {})

        if not child1_data or not child2_data:
            print(f"Object data not found for {child1_id} or {child2_id}")
            return None

        # Extract object names
        child1_name = child1_data.get("name", "unknown object")
        child2_name = child2_data.get("name", "unknown object")

        # Create prompt for VLM to analyze the relation
        prompt = f"Describe the spatial relationship between the {child1_name} and the {child2_name} in this image."

        # Here you would use the VLM to get the relation
        # For now, we'll just return a placeholder
        relation = {
            "object1": child1_id,
            "object2": child2_id,
            "object1_name": child1_name,
            "object2_name": child2_name,
            "relation": "placeholder relation"
        }

        # Save relation to file
        relation_file = os.path.join(res_dir, f"relation_{child1_id}_{child2_id}.json")
        with open(relation_file, 'w') as f:
            json.dump(relation, f, indent=4)

        return relation

if __name__ == "__main__":
    # get dataset dir and src image dir from argument
    parser = argparse.ArgumentParser(description="Generate relations between objects in images")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to the dataset JSON file")
    parser.add_argument("--src_image_dir", type=str, required=True, help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, default="relations", help="Directory to save relation results")

    args = parser.parse_args()

    # Create generator
    generator = VLMRelationGenerator(args.dataset_dir, args.src_image_dir)

    # Load dataset
    dataset = generator.load_dataset()
    if not dataset:
        print("Failed to load dataset. Exiting.")
        exit(1)
