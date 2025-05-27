from pixtral_utils.image_process import process_image_for_description

def instance_description_msg(image_path, mask_path):
    processed_image = process_image_for_description(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this highlighted object in the image"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }
    return message


def part_description_msg(image_path, mask_path, parent_description):
    processed_image = process_image_for_description(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Describe this highlighted part in the image, given that it is a part of a {parent_description}."},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }
    return message
