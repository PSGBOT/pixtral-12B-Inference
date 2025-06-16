import os
import shutil
import json
from vlm_utils.image_process import get_part_center

def json_output(key, key_to_sample_dir, output_dir, src_img_path, pair, image_id, instance_seg, kinematic_desc, sample_counter):
    if key not in key_to_sample_dir:
        sample_dir = os.path.join(
            output_dir,
            f"Sample_{sample_counter}",
        )
        os.makedirs(sample_dir, exist_ok=True)
        key_to_sample_dir[key] = sample_dir

        if os.path.exists(src_img_path):
            shutil.copy(
                src_img_path,
                os.path.join(sample_dir, "src_img.png"),
            )

        sample_counter += 1
    else:
        sample_dir = key_to_sample_dir[key]

    for mask_path in [pair[0], pair[1]]:
        if os.path.exists(mask_path):
            shutil.copy(
                mask_path,
                os.path.join(
                    sample_dir,
                    os.path.basename(mask_path),
                ),
            )

    config_path = os.path.join(
        sample_dir, "config.json"
    )
    config_data = {
        "part center": {},
        "kinematic relation": [],
    }

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError:
                pass

    for part_path in [pair[0], pair[1]]:
        part_name = os.path.splitext(
            os.path.basename(part_path)
        )[0]
        if part_name not in config_data["part center"]:
            center = get_part_center(
                image_id, instance_seg, part_path
            )
            if center:
                config_data["part center"][
                    part_name
                ] = center

    relation_entry = [
        os.path.splitext(os.path.basename(pair[0]))[0],
        os.path.splitext(os.path.basename(pair[1]))[0],
        kinematic_desc,
    ]
    config_data["kinematic relation"].append(
        relation_entry
    )

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    print(f"Updated {config_path}")