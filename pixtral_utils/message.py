from pixtral_utils.image_process import process_image_for_description, encode_image


def instance_description_msg(image_path, mask_path, debug=True):
    processed_image = process_image_for_description(
        image_path, mask_path, debug=debug, mask_level=0.75
    )

    # Create the message structure for the API
    message = {
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
    return [message]


def parse_description_msg(msg):
    message = [
        {"role": "system", "content": "Extract the description information."},
        {
            "role": "user",
            "content": msg,
        },
    ]
    return message


def part_relation_msg_for_KAF(
    image_path, mask_a, mask_b, parent_description, debug=True
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
        image_path, mask_a, debug=debug, mask_level=0.75
    )

    # Process the image with the second mask
    processed_image_b = process_image_for_description(
        image_path, mask_b, debug=debug, mask_level=0.75
    )

    # Create the prompt message structure for the API
    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""I will show you two images of a {parent_description}. In each image, a different part is highlighted in green.

First image: Look at the first highlighted part.
Second image: Look at the second highlighted part.

Please analyze the kinematic relationship between these two highlighted parts carefully:

1. First, identify each highlighted part by its specific name.

2. Describe in detail how these parts are physically connected to each other, using one or more of the following kinematic relationship terms:
   - Fixed/Rigid connection: Parts are firmly attached with no relative movement
   - Revolute joint: Parts rotate relative to each other around a single axis
   - Prismatic joint: Parts slide linearly relative to each other along a single axis
   - Cylindrical joint: Parts can both rotate and slide along the same axis
   - Spherical joint: Parts can rotate around a common point in any direction
   - Planar joint: Parts can translate in two dimensions and rotate around one axis
   - Supported: One part bears the weight of the other without rigid connection
   - Unrelated: Parts have no direct physical connection

3. If applicable, specify the axis or direction of movement (vertical, horizontal, etc.).

4. Explain whether the connection allows for controlled movement or is designed to be static.

5. If you can determine the purpose of this specific connection in the overall function of the {parent_description}, briefly explain it.

Format your response as:
Part 1: [name of first highlighted part]
Part 2: [name of second highlighted part]
Kinematic Relationship: [detailed description using the terminology above]
Movement Axis/Direction: [if applicable]
Purpose of Connection: [functional explanation]
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

    return message


def part_description_msg(image_path, mask_path, parent_description, debug=True):
    processed_image = process_image_for_description(image_path, mask_path, debug=debug)

    # Create the message structure for the API
    message = {
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
    return message
