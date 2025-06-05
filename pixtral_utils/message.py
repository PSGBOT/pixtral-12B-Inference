from pixtral_utils.image_process import process_image_for_description, encode_image


def instance_description_msg(image_path, mask_path, debug=True):
    processed_image = process_image_for_description(
        image_path, mask_path, debug=debug, crop=True, bbox=[641,343,754,453],padding_box=[-10,-10,10,10]
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


def part_relation_msg(image_path, mask_a, mask_b, parent_description):
    pass


def part_description_msg(image_path, mask_path, parent_description, debug=True):
    processed_image = process_image_for_description(image_path, mask_path, debug=debug,crop=True, bbox=[641,343,754,453],padding_box=[-10,-10,10,10])

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
