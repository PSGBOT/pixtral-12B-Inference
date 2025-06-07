from os.path import isfile
import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral
import json
import argparse
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import numpy as np
import re
import time
import random
from config import VLM_SETTINGS, LLM_SETTINGS
from pixtral_utils.output_structure import Instance, Part, KinematicRelationship
from pixtral_utils.message import crop_config


class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir, ouput_dir):
        """ EFFECT:
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
        self.output_dir = ouput_dir

        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            print("Error: MISTRAL_API_KEY environment variable not set.")
            exit(1)
        # Get model settings from config
        self.vlm = VLM_SETTINGS["model_name"]
        self.vlm_max_tokens = VLM_SETTINGS["max_tokens"]
        self.vlm_temperature = VLM_SETTINGS["temperature"]
        self.llm = LLM_SETTINGS["model_name"]
        self.llm_max_tokens = LLM_SETTINGS["max_tokens"]
        self.llm_temperature = LLM_SETTINGS["temperature"]
        print(f"Using model: {self.vlm}")
        # Initialize the Mistral client
        self.client = Mistral(api_key=api_key)

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

    def infer_vlm(self, msg, response_format=None):
        max_retries = 5
        base_delay = 2  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                if response_format == None:
                    chat_response = self.client.chat.complete(
                        model=self.vlm,
                        messages=msg,
                        max_tokens=self.vlm_max_tokens,
                        temperature=self.vlm_temperature,
                    )
                    return {"response": chat_response.choices[0].message.content}
                else:
                    chat_response = self.client.chat.parse(
                        model=self.vlm,
                        messages=msg,
                        response_format=response_format,
                        max_tokens=self.vlm_max_tokens,
                        temperature=self.vlm_temperature,
                    )
                    return json.loads(chat_response.choices[0].message.content)

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
                    raise

    def infer_llm(self, msg, response_format=None):
        """EFFECT: Generate the llm responses for joints between a pair"""
        max_retries = 5
        base_delay = 2  # Base delay in seconds

        for attempt in range(max_retries):
            try:
                if response_format == None:
                    chat_response = self.client.chat.complete(
                        model=self.llm,
                        messages=msg,
                        max_tokens=self.llm_max_tokens,
                        temperature=self.llm_temperature,
                    )
                    return {"response": chat_response.choices[0].message.content}
                else:
                    chat_response = self.client.chat.parse(
                        model=self.llm,
                        messages=msg,
                        response_format=response_format,
                        max_tokens=self.llm_max_tokens,
                        temperature=self.llm_temperature,
                    )
                    return json.loads(chat_response.choices[0].message.content)

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

    def generate_relation(self):
        relations_store = defaultdict(lambda: defaultdict(list))
        dump = bool(self.output_dir)
        for image_id in self.part_seg_dataset:
            image_res = self.part_seg_dataset[image_id]["masks"]
            for instance_seg in image_res:
                if "children" in image_res[instance_seg]:
                    p_mask_dir = os.path.join(
                        os.path.split(self.dataset_dir)[0],
                        image_id,
                        image_res[instance_seg]["path"],
                    )

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
                            debug=False,
                        )

                        # first generate dense description
                        instance_desc = self.infer_vlm(msg)
                        # then use llm to parse the description into json
                        structured_desc = self.infer_llm(
                            vlm_message.parse_instance_description_msg(
                                instance_desc["response"]
                            ),
                            response_format=Instance,
                        )
                        # process validity
                        if structured_desc["valid"] == "Yes":
                            structured_desc["valid"] = True
                        else:
                            structured_desc = {"valid": False}
                        self.part_seg_dataset[image_id]["masks"][instance_seg][
                            "description"
                        ] = structured_desc
                        print(structured_desc)
                    else:
                        print("existing description, skipping vlm")

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
                                mask1 = Image.open(pair[0]).convert("RGBA")
                                mask2 = Image.open(pair[1]).convert("RGBA")

                                split_width = 4
                                split_color = "white"
                                total_w = mask1.width + split_width + mask2.width
                                max_h   = max(mask1.height, mask2.height)

                                combined = Image.new("RGBA", (total_w, max_h), "WHITE")

                                combined.paste(mask1, (0, 0))

                                draw = ImageDraw.Draw(combined)
                                draw.rectangle(
                                    [mask1.width, 0, mask1.width + split_width, max_h],
                                    fill=split_color
                                )
                                combined.paste(mask2, (mask1.width + split_width, 0))
                                msg = vlm_message.part_relation_msg_for_KAF(
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
                                    debug=False,
                                )
                                kinematic_desc = self.infer_vlm(msg)
                                print(kinematic_desc)
                                structured_kinematic_desc = self.infer_llm(
                                    vlm_message.parse_part_relation_msg(
                                        kinematic_desc["response"]
                                    ),
                                    KinematicRelationship,
                                )
                                print(structured_kinematic_desc)
                                relations_store[image_id][instance_seg].append({
                                    "parts": [os.path.basename(pair[0]), os.path.basename(pair[1])],
                                    "relation": structured_kinematic_desc,
                                })
                                caption = str(structured_kinematic_desc)
                                font = ImageFont.load_default()
                                pad = 8

                                draw = ImageDraw.Draw(combined)
                                text_x = pad
                                text_y = combined.height + pad
                                draw.text((text_x, text_y), caption, fill="white", font=font)
                                combined.show()
        if dump:
            for img_id, inst_dict in relations_store.items():
                for inst_seg, rel_list in inst_dict.items():
                    out_dir = os.path.join(self.output_dir, img_id, inst_seg)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, "relations.json")
                    with open(out_path, "w") as f:
                        json.dump(rel_list, f, indent=4)
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
