from pixtral_utils.image_process import process_image_for_description


def instance_description_msg(image_path, mask_path):
    processed_image = process_image_for_description(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": "Please introduce the name and usage of the highlighted object in the image. This is the image:"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }
    return message

def part_relation_msg(image_path, mask_path, parent_description):
    pass

def part_description_msg(image_path, mask_path, parent_description):
    processed_image = process_image_for_description(image_path, mask_path)

    # Create the message structure for the API
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"The highlighted part in the image is a part of a {parent_description}. Please introduce the name and purpose of this part. If its purpose is too subtle, you can ignore the request of introducing its purpose. If there is any text on this component, also output the text. This is the image:"},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{processed_image}"}
        ]
    }
    return message
