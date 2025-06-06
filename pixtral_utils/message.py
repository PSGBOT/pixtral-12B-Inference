from pixtral_utils.image_process import (
    process_image_for_description,
    encode_image,
    crop_config,
)


def instance_description_msg(
    image_path,
    mask_path,
    debug=True,
    crop_config=crop_config(),
):
    processed_image = process_image_for_description(
        image_path,
        mask_path,
        crop_config=crop_config,
        debug=debug,
    )

    # Create the message structure for the API
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": """Focus on the area highlighted in green in the image.

Step 1: Determine if the highlighted area represents a distinct, identifiable object or instance:
- If the highlighted area is clearly a distinct object, proceed to Step 2.
- If the highlighted area is abstract, ambiguous, or you cannot confidently identify it as a specific object (e.g., part of background, texture, partial view), respond with "Valid: No".

Step 2: If the highlighted area is a distinct object, provide:
1. The specific name of the object (be precise and use technical terms when appropriate)
2. The primary function or purpose of this object
3. Any notable features visible in the highlighted area (no color description)
4. If there is text visible on the object, include what it says

Remember, if you're uncertain about the highlighted area being a distinct object, respond only with "Valid: No".
""",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{processed_image}",
            },
        ],
    }
    return [user_message]


def parse_instance_description_msg(msg):
    message = [
        {"role": "system", "content": "Extract the description information."},
        {
            "role": "user",
            "content": msg,
        },
    ]
    return message


def part_relation_msg_for_KAF(
    image_path,
    mask_a,
    mask_b,
    parent_description,
    crop_config=crop_config(),
    debug=True,
):
    """
    Create a message to query the kinematic relationship between two parts of an object.

    Args:
        image_path (str): Path to the original image
        mask_a (str): Path to the mask for the first part
        mask_b (str): Path to the mask for the second part
        parent_description (str): Description of the parent object
        debug (bool): Whether to display debug images

    Returns:
        dict: Message structure for the API
    """
    # Process the image with the first mask
    processed_image_a = process_image_for_description(
        image_path,
        mask_a,
        mask_level=0.15,
        highlight_level=0.6,
        crop_config=crop_config,
        debug=debug,
    )

    # Process the image with the second mask
    processed_image_b = process_image_for_description(
        image_path,
        mask_b,
        mask_level=0.15,
        highlight_level=0.6,
        crop_config=crop_config,
        debug=debug,
    )

    # Create the prompt message structure for the API
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""You are an expert mechanical engineer specializing in kinematic analysis of mechanical systems. I will show you two images of a {parent_description}. In each image, a different part is highlighted in green.

Image 0: The first highlighted part (Part 0).
Image 1: The second highlighted part (Part 1).

Your task is to analyze the precise kinematic relationship between these two highlighted parts (no other parts should be involved):

1. Identify each highlighted part with its technical name.

2. Determine all possible types of kinematic joint or connection between these parts,  one or more of these standard mechanical engineering terms:
   - fixed: Parts are firmly attached with no relative movement
   - revolute: Parts rotate relative to each other around a single axis
   - prismatic: Parts slide linearly relative to each other along a single axis
   - cylindrical: Parts can both rotate and slide along the same axis
   - spherical: Parts can rotate around a common point in any direction
   - planar: Parts can translate in two dimensions and rotate around one axis
   - press: One part can be pressed into another with spring resistance
   - supported: One part bears the weight of the other without rigid connection
   - unrelated: Parts are not directly connected or attached to each other

3. Specify the all possible axis or direction of movement, using the following standard terms:
    - vertical
    - horizontal
    - radial
    - axial

4. Determine whether the connection is:
   - static: Designed to prevent movement
   - controlled: Allows specific, limited movement
   - free: Allows unrestricted movement within the joint's degrees of freedom

5. Explain the functional purpose of this specific connection in the overall operation of the {parent_description}.

6. Identify which part serves as the kinematic root (the more fixed/stable part that the other part moves relative to). Use the following criteria to determine the root:
   - The part that remains stationary while the other part moves
   - The part that is attached to the main structure or frame
   - The part that constrains or guides the movement of the other part
   - The part that would typically be considered the "base" in engineering terms

   After analysis, specify exactly one of these answers:
   - "0" if the part in image 0 is the kinematic root
   - "1" if the part in image 1 is the kinematic root
   - "neither" if both parts move equally relative to each other or if they are both attached to a third part that serves as the actual root

If multiple joint types exist between these parts, list each one separately using the format above.
""",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{processed_image_a}",
            },
            {"type": "text", "text": "Second image with different highlighted part:"},
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{processed_image_b}",
            },
        ],
    }

    return [user_message]


def parse_part_relation_msg(msg):
    message = [
        {
            "role": "system",
            "content": """Extract the kinematic relationship information between the parts described in the message.

Your task is to identify and structure the kinematic joint information according to these rules:

1. Identify valid kinematic joint types in the text. Only use these standard terms:
   - fixed
   - revolute
   - prismatic
   - cylindrical
   - spherical
   - planar
   - press
   - supported
   - unrelated

2. For each joint identified, extract:
   - joint_type: The type of joint from the standard list above
   - joint_movement_axis: The axis or direction of movement (e.g., vertical, horizontal, radial)
   - is_static: Whether the joint is static or allows movement
   - purpose: The functional purpose of this joint

3. If multiple joint types exist between the parts, include each as a separate object in the kinematic_joints array.

4. For the root part, use part id (`0` or `1`), `0` for the first image, `1` for the second image.

Be precise and only extract information that is explicitly stated in the message.""",
        },
        {
            "role": "user",
            "content": msg,
        },
    ]
    return message


# deprecated
def part_description_msg(image_path, mask_path, parent_description, debug=True):
    processed_image = process_image_for_description(
        image_path,
        mask_path,
        debug=debug,
        crop=True,
        bbox=[641, 343, 754, 453],
        padding_box=[-10, -10, 10, 10],
    )

    # Create the message structure for the API
    user_message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"The highlighted (as green) part in the image is a part of a {parent_description}. Please introduce the name and purpose of this part. If its purpose is too subtle, you can ignore the request of introducing its purpose. If there is any text on this component, also output the text. This is the image:",
            },
            {
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{processed_image}",
            },
        ],
    }
    return [user_message]
