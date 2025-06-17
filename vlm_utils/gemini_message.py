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
        f"""You are an expert mechanical engineer specializing in kinematic analysis of mechanical systems. I will show you two images showing two parts of a {parent_description}. In each image, the interested part is highlighted in green.
""",
        "This is the image of part 0:",
        types.Part.from_bytes(
            data=processed_image_a,
            mime_type="image/jpeg",
        ),
        "This is the image of part 1:",
        types.Part.from_bytes(
            data=processed_image_b,
            mime_type="image/jpeg",
        ),
        """

Your task is to analyze the precise kinematic relationship between these two highlighted parts (no other parts out of the highlighted area should be involved):

1. Describe each highlighted part briefly

2. Determine the function of each parts, using the one or more of these standard terms:
- handle: a part which is designed to hold or carry something
- housing: a protective enclosure for components
- support: a part designed to bear weight or provide stability
- frame: a rigid structure that provides support or a framework for something
- button: a small knob or disc that is pushed or pressed to operate something, nozzle is also a button
- wheel: a circular object that revolves on an axle and is fixed below a vehicle or other object to enable it to move easily over the ground
- display: presenting visual information (text or image)
- cover: a lid or other removable top for a container
- plug: a device for making an electrical connection, typically having two or three pins that are inserted into sockets
- port: an opening in the surface of an electronic device through which another device can be connected
- door: an opening in the surface of a structure that allows entry or exit
- container: a receptacle for holding or containing something
- other: something that does not fit any of the above


3. Determine possible types of kinematic joint or connection between these parts, using one or more of these standard mechanical engineering terms:
   - fixed: parts are firmly attached with no relative movement
   - revolute: parts rotate relative to each other around a single axis
   - prismatic: parts slide linearly relative to each other along a single axis
   - spherical: parts can rotate around a common point in any direction
   - supported: One part bears the weight of the other without rigid connection
   - flexible: parts are connected with a flexible connection, such as spring, cable or fabric
   - unrelated: parts are not directly connected or attached to each other
   - unknown: does not fit any of the above

4. Determine whether the connection is:
   - static: Designed to prevent movement
   - controlled: Allows specific, limited movement
   - free: Allows unrestricted movement within the joint's degrees of freedom

5. Identify which part serves as the kinematic root (the more fixed/stable part that the other part moves relative to). Use the following criteria to determine the root:
   - The part that remains stationary while the other part moves
   - The part that is attached to the main structure or frame
   - The part that constrains or guides the movement of the other part
   - The part that would typically be considered the "base" in engineering terms

   After analysis, specify exactly one of these answers:
   - "0" if the part in first image is the kinematic root
   - "1" if the part in second image is the kinematic root
   - "neither" if both parts move equally relative to each other or if they are both attached to a third part that serves as the actual root

If multiple joint types exist between these parts, list each one separately using the format above.
""",
    ]

    vis_a = None
    vis_b = None
    if debug:
        vis_a = Image.open(BytesIO(base64.b64decode(processed_image_a)))
        vis_b = Image.open(BytesIO(base64.b64decode(processed_image_b)))

    return user_message, (vis_a, vis_b) if debug else None
