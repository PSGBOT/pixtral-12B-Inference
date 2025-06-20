import json
import os
import networkx as nx
import argparse
from nx_utils.build_nx import read_rel_as_nx
from nx_utils.visualize import show_graph

from nx_utils.validate_graph import (
    detect_conflict_kr,
    detect_invalid_kr,
    detect_redundancy_kr,
    detect_cyclic_kr,
)

PSR_KR_CAT = [
    "unknown",
    "fixed",
    "revolute-free",
    "revolute-controlled",
    "revolute-static",
    "prismatic-free",
    "prismatic-controlled",
    "prismatic-static",
    "spherical-free",
    "spherical-controlled",
    "spherical-static",
    "supported",
    "flexible",
    "unrelated",
]
PSR_FUNC_CAT = [
    "other",
    "handle",
    "housing",
    "support",
    "frame",
    "button",
    "wheel",
    "display",
    "cover",
    "plug",
    "port",
    "door",
    "container",
]


def _parse_relations(relations):
    # relations is expected to be a dictionary, e.g., {'type': 'revolute-free', 'root_index': 0}
    joint_type = relations.get("joint_type", "unknown")
    controllable = relations.get("controllable")

    if joint_type in ["fixed", "unrelated", "supported", "flexible"]:
        relation_type = joint_type
    elif controllable:
        relation_type = f"{joint_type}-{controllable}"
    else:
        relation_type = joint_type  # Fallback, though should be covered by above

    root = int(relations.get("root", 0))  # Convert root to integer
    return relation_type, root


def prune_kinematic_relation(kr_dict):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prune the kinematic relation generated by VLM"
    )
    parser.add_argument(
        "--dataset_dir", type=str, required=True, help="Path to the PSR dataset"
    )

    args = parser.parse_args()

    sample_list = os.listdir(args.dataset_dir)
    for sample_name in sample_list:
        print(f"Processing {sample_name}...")
        sample_dir = os.path.join(args.dataset_dir, sample_name)
        config_path = os.path.join(sample_dir, "config.json")
        mask_path = os.path.join(sample_dir, "mask0.png")
        src_img_path = os.path.join(sample_dir, "src_img.png")
        with open(config_path, "r") as f:
            psr_dict = json.load(f)
            kr_list = psr_dict["kinematic relation"]
            pos_dict = psr_dict["part center"]
        G = read_rel_as_nx(kr_list, pos_dict)
        show_graph(G, src_img_path, mask_path)
