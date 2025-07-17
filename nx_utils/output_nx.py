import json
import os
import networkx as nx


def integrate_kr_list(kr_list, new_kr):
    """
    new_kr: [part0_str, part1_str, kj_list]
    kr_list: [kr0, kr2, ....]
    if new_kr's' part0_str, part1_str match the kr in kr_list, then merge the new_kr's kj_list into the kr, or add the new_kr to kr_list
    """

    for kr in kr_list:
        if kr[0] == new_kr[0] and kr[1] == new_kr[1]:
            # Merge kj_list if parts match
            kr[2]["kinematic_joints"].extend(new_kr[2]["kinematic_joints"])
            return kr_list
    # If no match found, add the new_kr to the list
    kr_list.append(new_kr)
    return kr_list


def create_new_config_json(
    sample_dir, G, kr_list, pos_dict, new_config_filename="new_config.json"
):
    config_path = os.path.join(sample_dir, new_config_filename)
    config_data = {"part center": {}, "kinematic relation": []}

    for node in G.nodes:
        part_name = str(node)  # Convert node to string for consistency
        if part_name not in config_data["part center"]:
            # Get the part center from the pos_dict (if available)
            center = pos_dict.get(part_name, [0, 0])  # Default to [0, 0] if not found
            config_data["part center"][part_name] = center

    for p0, p1, data in G.edges(data=True):
        root = data.get("root", 0)
        if root == "0":
            part0 = str(p1)
            part1 = str(p0)
        else:
            part0 = str(p0)
            part1 = str(p1)
        joint_type = data.get("joint_type", "unknown")
        controllable = data.get("controllable", "static")
        root = data.get("root", 0)
        part0_func = data.get("part0_function", [])
        part1_func = data.get("part1_function", [])
        part0_desc = data.get("part0_desc", "")
        part1_desc = data.get("part1_desc", "")
        kr = [
            part0,
            part1,
            {
                "part0_desc": part0_desc,
                "part1_desc": part1_desc,
                "part0_function": part0_func,
                "part1_function": part1_func,
                "kinematic_joints": [
                    {
                        "joint_type": joint_type,
                        "controllable": controllable,
                        "root": root,
                    }
                ],
            },
        ]
        config_data["kinematic relation"] = integrate_kr_list(
            config_data["kinematic relation"], kr
        )

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"Created new {config_path}")


# networkX包
# 把原本的json文件用原本函数变成G，然后用这个函数测一下relation是否相同
