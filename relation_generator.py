from os.path import isfile
import pixtral_utils.message as vlm_message
import os
from mistralai import Mistral
import json
import argparse
from PIL import Image
from collections import defaultdict
import numpy as np
import time
import random
from config import VLM_SETTINGS, LLM_SETTINGS
from pixtral_utils.output_structure import Instance, Part


class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir, ouput_dir):
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
        merged = defaultdict(list)
        for d in dicts:
            for k, v in d.items():
                merged[k].extend(v)  # 合并 list
        return dict(merged)

    def load_dataset(self):
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
                            debug=False,
                        )

                        # first generate dense description
                        instance_desc = self.infer_vlm(msg)
                        # then use llm to parse the description into json
                        structures_desc = self.infer_llm(
                            vlm_message.parse_description_msg(
                                instance_desc["response"]
                            ),
                            response_format=Instance,
                        )
                        # process validity
                        if structures_desc["valid"] == "Yes":
                            structures_desc["valid"] = True
                        else:
                            structures_desc = {"valid": False}
                        self.part_seg_dataset[image_id]["masks"][instance_seg][
                            "description"
                        ] = structures_desc
                    else:
                        print("existing description, skipping vlm")

                    # process pairs
                    if self.part_seg_dataset[image_id]["masks"][instance_seg][
                        "description"
                    ]["valid"]:
                        src_img_path = os.path.join(
                            self.src_image_dir, f"{image_id}.png"
                        )
                        for pair in pairs[p_mask_dir]:
                            msg = vlm_message.part_relation_msg_for_KAF(
                                src_img_path,
                                pair[0],
                                pair[1],
                                self.part_seg_dataset[image_id]["masks"][instance_seg][
                                    "description"
                                ]["name"],
                            )
        # store the description(valuable)
        with open(
            os.path.split(self.dataset_dir)[0] + "_with_description.json", "w"
        ) as f:
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
