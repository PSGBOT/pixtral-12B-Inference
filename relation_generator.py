import os
import json
import argparse
import shutil  # Added for file operations
from tqdm import tqdm
from collections import defaultdict
from config import FLASH_VLM_SETTINGS, LLM_SETTINGS, SOTA_VLM_SETTINGS
from vlm_utils.image_process import combined_image_present
from vlm_utils.vlm_service import VLMService
from vlm_utils.json_output import json_output


class VLMRelationGenerator:
    def __init__(self, dataset_dir, src_image_dir, output_dir, processed_dataset_dir):
        """EFFECT:
            Initialize the vlm relation generator.
        INPUT:
            dataset_dir: dataset description json file
            src_image_dir: dataset source image directory
            output_dir: optional, output directort for the KAF json files
            processed_dataset_dir: optional, directory to move processed images
        """
        self.dataset_dir = dataset_dir
        self.src_image_dir = src_image_dir
        self.dataset = {}
        self.part_seg_dataset = {}
        self.key_to_sample_dir = {}
        self.output_dir = output_dir
        self.processed_dataset_dir = (
            processed_dataset_dir  # Added processed_dataset_dir
        )

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

        # get the number of exist outputs in output_dir
        self.output_count = len(os.listdir(output_dir))
        print(f"Exist {self.output_count} samples")
        self.sample_counter = self.output_count

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

            # backup the dataset json as *.bkp if the backup doesn't exist
            backup_path = self.dataset_dir + ".bkp"
            if not os.path.exists(backup_path):
                shutil.copy2(self.dataset_dir, backup_path)
                print(f"Created dataset backup: {backup_path}")

            return True

        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False

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

    def generate_relation(self, debug=False):
        root = os.path.dirname(self.dataset_dir)
        dump = bool(self.output_dir)
        for image_id in tqdm(self.part_seg_dataset, desc="Processing images"):
            try:
                if "processed" in self.part_seg_dataset[image_id]:
                    print(f"skip image {image_id} already processed")
                    continue
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
                            vlm_service = VLMService(
                                "GEMINI"
                            )  ## FIXME: "MISTRAL" = pixtral 12B, "GEMINI" = gemini
                            instance_desc = vlm_service.instance_description(
                                self.src_image_dir,
                                image_id,
                                p_mask_dir,
                                image_res[instance_seg]["bbox"],
                            )
                            self.part_seg_dataset[image_id]["masks"][instance_seg][
                                "description"
                            ] = instance_desc
                            print("INSTANCE DESCRIPTION: ")
                            print(instance_desc)
                        else:
                            print(
                                "INSTANCE DESCRIPTION: existing description, skipping vlm"
                            )

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

                            vlm_service = VLMService(
                                "GEMINI"
                            )  ## FIXME: "MISTRAL" = pixtral 12B, "GEMINI" = gemini
                            # Process all pairs for matching keys
                            for key in matching_keys:  # key is the directrory
                                for pair in pairs[key]:
                                    kinematic_desc, vis_img = (
                                        vlm_service.kinematic_description(
                                            src_img_path,
                                            pair,
                                            self.part_seg_dataset[image_id]["masks"][
                                                instance_seg
                                            ]["description"]["name"],
                                            image_res[instance_seg]["bbox"],
                                            debug,
                                        )
                                    )
                                    print("KINE DESC: ")
                                    print(kinematic_desc)
                                    if dump:
                                        json_output(
                                            key,
                                            self.key_to_sample_dir,
                                            self.output_dir,
                                            src_img_path,
                                            pair,
                                            image_id,
                                            instance_seg,
                                            kinematic_desc,
                                            self.sample_counter,
                                        )
                                    if debug:
                                        combined_image_present(vis_img, kinematic_desc)

                                if dump:
                                    self.sample_counter += 1

                self.part_seg_dataset[image_id]["processed"] = True
                # store the description(valuable)
                with open(self.dataset_dir, "w") as f:
                    json.dump(self.part_seg_dataset, f, indent=4)
            except Exception as e:
                print(f"Error processing image {image_id}: {e}")
                break


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
    parser.add_argument(  # Added new argument for processed images
        "--processed_dataset_dir",
        type=str,
        default=None,
        help="Directory to move processed source images after processing",
    )

    args = parser.parse_args()

    # Create generator
    generator = VLMRelationGenerator(
        args.dataset_dir,
        args.src_image_dir,
        args.output_dir,
        args.processed_dataset_dir,
    )

    # Load dataset
    generator.load_dataset()

    generator.generate_relation(debug=False)
