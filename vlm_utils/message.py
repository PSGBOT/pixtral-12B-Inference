from vlm_utils.image_process import (
    process_image_for_description,
    encode_image,
    crop_config,
)
from google.genai import types
import base64
from io import BytesIO
from PIL import Image


def instance_description_msg(
    image_path,
    mask_path,
    debug=True,
    crop_config=crop_config(),
):
    processed_image = process_image_for_description(
        image_path,
        mask_path,
        mask_level=0.8,
        crop_config=crop_config,
        debug=debug,
    )

    # Create the message structure for the API
    user_message = (
        [
            """Focus only on the area highlighted in green in the image.

Step 1: Determine if the highlighted area represents a distinct, identifiable foreground instance:
- If the highlighted area is clearly a distinct foreground instance that is interactable, proceed to Step 2.
- Else, respond with "Valid: No".

Step 2: If the highlighted area is a distinct object, provide:
1. The specific name of the object (be precise and use technical terms when appropriate)
2. The primary function or purpose of this object
3. Any notable features visible in the highlighted area (no color description)
4. If there is text visible on the object, include what it says

Remember, if you're uncertain about the highlighted area being a distinct object, respond only with "Valid: No".
""",
            types.Part.from_bytes(
                data=processed_image,
                mime_type="image/jpeg",
            ),
        ],
    )
    return user_message


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
        debug=False,  # NOTE: Change this to debug if separated images are needed
    )

    # Process the image with the second mask
    processed_image_b = process_image_for_description(
        image_path,
        mask_b,
        mask_level=0.15,
        highlight_level=0.6,
        crop_config=crop_config,
        debug=False,  # NOTE: Change this to debug if separated images are needed
    )

    # Create the prompt message structure for the API
    user_message = [
        f"""You are an expert mechanical engineer specializing in kinematic analysis of mechanical systems. I will show you two images showing two components of a {parent_description}. In each image, the interested component is highlighted in green.

Your task is to analyze the precise kinematic relationship between these two highlighted components (no other components out of the highlighted area should be involved):

1. describe each highlighted component briefly

2. Determine all possible types of kinematic joint or connection between these components, using one or more of these standard mechanical engineering terms:
   - fixed: Components are firmly attached with no relative movement
   - revolute: Components rotate relative to each other around a single axis
   - prismatic: Components slide linearly relative to each other along a single axis
   - spherical: Components can rotate around a common point in any direction
   - supported: One component bears the weight of the other without rigid connection
   - unrelated: Components are not directly connected or attached to each other

4. Determine whether the connection is:
   - static: Designed to prevent movement
   - controlled: Allows specific, limited movement
   - free: Allows unrestricted movement within the joint's degrees of freedom

5. Explain the functional purpose of this specific connection in the overall operation of the {parent_description}.

6. Identify which component serves as the kinematic root (the more fixed/stable part that the other part moves relative to). Use the following criteria to determine the root:
   - The component that remains stationary while the other part moves
   - The component that is attached to the main structure or frame
   - The component that constrains or guides the movement of the other part
   - The component that would typically be considered the "base" in engineering terms

   After analysis, specify exactly one of these answers:
   - "0" if the component in first image is the kinematic root
   - "1" if the component in second image is the kinematic root
   - "neither" if both components move equally relative to each other or if they are both attached to a third component that serves as the actual root

If multiple joint types exist between these components, list each one separately using the format above.
""",
        types.Part.from_bytes(
            data=processed_image_a,
            mime_type="image/jpeg",
        ),
        types.Part.from_bytes(
            data=processed_image_a,
            mime_type="image/jpeg",
        ),
    ]

    vis_a = None
    vis_b = None
    if debug:
        vis_a = Image.open(BytesIO(base64.b64decode(processed_image_a)))
        vis_b = Image.open(BytesIO(base64.b64decode(processed_image_b)))

    return user_message, (vis_a, vis_b) if debug else None
