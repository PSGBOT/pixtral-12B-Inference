from os.path import isfile
import vlm_utils.message as vlm_message
import os
import shutil
from google import genai
import json
import cv2
import argparse
from collections import defaultdict
import time
import random
from config import FLASH_VLM_SETTINGS, LLM_SETTINGS, SOTA_VLM_SETTINGS
from vlm_utils.output_structure import Instance, Part, KinematicRelationship
from vlm_utils.message import crop_config
from vlm_utils.image_process import combined_image_present
import vlm_utils.message as vlm_message

class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir, ouput_dir):
        """EFFECT:
            Initialize the vlm relation generator.
        INPUT:
            dataset_dir: dataset description json file
            src_image_dir: dataset source image directory
            output_dir: optional, output directort for the KAF json files
        """
        self.dataset_dir = dataset_dir
        self.src_image_dir = src_image_dir
        self.dataset = {}
        self.part_seg_dataset = {}
        self.key_to_sample_dir = {}
        self.output_dir = ouput_dir
        self.sample_counter = 0

        api_key = os.environ.get("GENAI_API_KEY")
        if not api_key:
            print("Error: MISTRAL_API_KEY environment variable not set.")
            exit(1)
        # Get model settings from config
        self.flash_vlm = FLASH_VLM_SETTINGS["model_name"]
        self.flash_vlm_max_tokens = FLASH_VLM_SETTINGS["max_tokens"]
        self.flash_vlm_temperature = FLASH_VLM_SETTINGS["temperature"]
        self.sota_vlm = SOTA_VLM_SETTINGS["model_name"]
        self.sota_vlm_max_tokens = SOTA_VLM_SETTINGS["max_tokens"]
        self.sota_vlm_temperature = SOTA_VLM_SETTINGS["temperature"]
        self.llm = LLM_SETTINGS["model_name"]
        self.llm_max_tokens = LLM_SETTINGS["max_tokens"]
        self.llm_temperature = LLM_SETTINGS["temperature"]
        print(f"Using model: {self.flash_vlm}")
        # Initialize the Mistral client
        self.client = genai.Client(api_key=api_key)

    def merge_dicts(self, dicts):
        """EFFECT: Merge multiple dicts into one"""
        merged = defaultdict(list)
        for d in dicts:
            for k, v in d.items():
                merged[k].extend(v)  # 合并 list
        return dict(merged)

    def load_dataset(self):
        """EFFECT: Load dataset description json into dataset"""
        try:
            if not os.path.exists(self.dataset_dir):
                print(f"Dataset file not found: {self.dataset_dir}")
                return None

            with open(self.dataset_dir, "r") as f:
                dataset = json.load(f)

            print(f"Loaded part segmentation dataset from {self.dataset_dir}")
            print(f"Dataset contains {len(dataset)} images")

            self.part_seg_dataset = dataset
            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

    def infer_vlm(self, msg, response_format=None, vlm=0):
        max_retries = 5
        base_delay = 2  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                if response_format == None:
                    chat_response = self.client.models.generate_content(
                        model=self.flash_vlm if vlm == 0 else self.sota_vlm,
                        contents=msg,
                    )
                    return json.loads(chat_response.text)
                else:
                    print("using format")
                    chat_response = self.client.models.generate_content(
                        model=self.flash_vlm if vlm == 0 else self.sota_vlm,
                        contents=msg,
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": response_format,
                        },
                    )
                    return json.loads(chat_response.text)

            except Exception as e:
                # Check if it's a rate limit error
                if (
                    "rate limit" in str(e).lower()
                    or "too many requests" in str(e).lower()
                ):
                    if attempt < max_retries - 1:  # Don't sleep on the last attempt
                        # Calculate exponential backoff with jitter
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"Rate limit exceeded. Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        print(
                            f"Failed after {max_retries} attempts due to rate limiting."
                        )
                        raise
                else:
                    # If it's not a rate limit error, re-raise the exception
                    print(f"API error: {e}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2**attempt) + random.uniform(0, 1)
                        print(
                            f"Retrying in {delay:.2f} seconds... (Attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise

    def generate_pairs(self, p_mask_dir, children_dir) -> dict:
        res = {}
        all_dir = os.listdir(children_dir)
        children_path = []
        for child in all_dir:
            child_path = os.path.join(children_dir, child)
            if os.path.isfile(child_path):
                children_path.append(child_path)
            else:
                res = self.merge_dicts(
                    [res, self.generate_pairs(child_path + ".png", child_path)]
                )

        res[p_mask_dir] = []

        for child_a_id in range(len(children_path)):
            for child_b_id in range(child_a_id + 1, len(children_path)):
                res[p_mask_dir].append(
                    (children_path[child_a_id], children_path[child_b_id])
                )
        return res

    def get_part_center(self, image_id, instance_id, part_mask_path):
        """EFFECT: Compute the centroid coordinates of a part's binary mask.
        INPUT:
            image_id: ID of the image (unused but kept for interface consistency)
            instance_id: ID of the instance (unused but kept for interface consistency)
            part_mask_path: Path to the part's binary mask image
        OUTPUT:
            (x, y) tuple of centroid coordinates if successful, None otherwise
        """
        try:
            if not isfile(part_mask_path):
                return None
            
            mask = cv2.imread(part_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                return None
                
            M = cv2.moments(mask)
            if M["m00"] == 0: 
                return None
                
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
            
        except ImportError:
            print("Error: OpenCV required for center calculation")
            return None
        except Exception as e:
            print(f"Error computing part center: {e}")
            return None

    def generate_relation(self):
        relations_store = defaultdict(lambda: defaultdict(list))
        root = os.path.dirname(self.dataset_dir)
        dump = bool(self.output_dir)
        for image_id in self.part_seg_dataset:
            image_res = self.part_seg_dataset[image_id]["masks"]
            for instance_seg in image_res:
                if "children" in image_res[instance_seg]:
                    seg = image_res[instance_seg]
                    p_mask_dir = os.path.join(root, image_id, seg["path"])


                    children_dir, _ = os.path.splitext(p_mask_dir)
                    pairs = self.generate_pairs(p_mask_dir, children_dir)

                    # generate description for parent instance
                    if "description" not in image_res[instance_seg]:
                        print("processing description for parent instance")
                        msg = vlm_message.instance_description_msg(
                            os.path.join(self.src_image_dir, f"{image_id}.png"),
                            p_mask_dir,
                            crop_config=crop_config(
                                True,
                                bbox=image_res[instance_seg]["bbox"],
                                padding_box=[-20, -20, 20, 20],
                            ),
                            debug=True,
                        )

                        # first generate dense description
                        instance_desc = self.infer_vlm(msg, Instance, vlm=1)
                        # process validity
                        if instance_desc["valid"] == "Yes":
                            instance_desc["valid"] = True
                        else:
                            instance_desc = {"valid": False}
                        self.part_seg_dataset[image_id]["masks"][instance_seg][
                            "description"
                        ] = instance_desc
                        print("INSTANCE DESCRIPTION: ")
                        print(instance_desc)
                    else:
                        print("INSTANCE DESCRIPTION: existing description, skipping vlm")

                    # process pairs
                    if self.part_seg_dataset[image_id]["masks"][instance_seg][
                        "description"
                    ]["valid"]:
                        src_img_path = os.path.join(
                            self.src_image_dir, f"{image_id}.png"
                        )
                        # Search for keys in the pairs dict that match or contain p_mask_dir
                        matching_keys = []
                        for key in pairs:
                            if os.path.splitext(p_mask_dir)[0] in key:
                                matching_keys.append(key)
                        print(matching_keys)

                        # Process all pairs for matching keys
                        for key in matching_keys:  # key is the directrory
                            for pair in pairs[key]:
                                msg, vis_img = vlm_message.part_relation_msg_for_KAF(
                                    src_img_path,
                                    pair[0],
                                    pair[1],
                                    self.part_seg_dataset[image_id]["masks"][
                                        instance_seg
                                    ]["description"]["name"],
                                    crop_config=crop_config(
                                        False,
                                        bbox=image_res[instance_seg]["bbox"],
                                        padding_box=[-20, -20, 20, 20],
                                    ),
                                    debug=True,
                                )
                                kinematic_desc = self.infer_vlm(
                                    msg, KinematicRelationship
                                )
                                print("KINE DESC: ")
                                print(kinematic_desc)
                                if dump:
                                    if key not in self.key_to_sample_dir:
                                        sample_dir = os.path.join(self.output_dir, f"Sample_{self.sample_counter}")
                                        os.makedirs(sample_dir, exist_ok=True)
                                        self.key_to_sample_dir[key] = sample_dir
                                        
                                        if os.path.exists(src_img_path):
                                            shutil.copy(src_img_path, os.path.join(sample_dir, "src_img.png"))
                                        
                                        self.sample_counter += 1
                                    else:
                                        sample_dir = self.key_to_sample_dir[key]
                                    
                                    for mask_path in [pair[0], pair[1], key]:
                                        if os.path.exists(mask_path):
                                            shutil.copy(mask_path, os.path.join(sample_dir, os.path.basename(mask_path)))

                                    config_path = os.path.join(sample_dir, "config.json")
                                    config_data = {"part center": {}, "kinematic relation": []}

                                    if os.path.exists(config_path):
                                        with open(config_path, 'r') as f:
                                            try:
                                                config_data = json.load(f)
                                            except json.JSONDecodeError:
                                                pass
                                    
                                    for part_path in [pair[0], pair[1]]:
                                        part_name = os.path.splitext(os.path.basename(part_path))[0]
                                        if part_name not in config_data["part center"]:
                                            center = self.get_part_center(image_id, instance_seg, part_path)
                                            if center:
                                                config_data["part center"][part_name] = center

                                    relation_entry = [
                                        os.path.basename(pair[0]), 
                                        os.path.basename(pair[1]),
                                        kinematic_desc
                                    ]
                                    config_data["kinematic relation"].append(relation_entry)

                                    with open(config_path, 'w') as f:
                                        json.dump(config_data, f, indent=4)
                                    
                                    print(f"Updated {config_path}")
                                combined_image_present(vis_img, kinematic_desc)
        
        # store the description(valuable)
        with open(self.dataset_dir, "w") as f:
            json.dump(self.part_seg_dataset, f, indent=4)


if __name__ == "__main__":
    # get dataset dir and src image dir from argument
    parser = argparse.ArgumentParser(
        description="Generate relations between objects in images"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--src_image_dir",
        type=str,
        required=True,
        help="Directory containing source images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="relations",
        help="Directory to save relation results",
    )

    args = parser.parse_args()

    # Create generator
    generator = VLMRelationGenerator(
        args.dataset_dir, args.src_image_dir, args.output_dir
    )

    # Load dataset
    generator.load_dataset()

    generator.generate_relation()
