# pixtral-12B-Inference
This repository contains Python code examples to interact with the PixTral 12B API.

## File Descriptions:
`generate.py`: simple script for generate object/part description from image using pixtral api.

## Setup
- Setup `your key` as environment variable `PIXTRAL_KEY`
- Download the sample dataset from feishu group `resources/Data/part_seg_dataset sample.zip` and extract it to a directory (DON'T put the dataset in this project directory)
- Specify the directory in `config/custom_cfg`

## Relation Generation
Related script: `relation_generator.py`

```bash
python relation_generator.py --dataset_dir "/home/cyl/Reconst/Data/Sample dataset/part_seg_dataset/part_seg_dataset_with_description.json" --src_image_dir "/home/cyl/Reconst/Data/Sample dataset/src_img"
```

### Process

#### Instance Description
First for each root level mask, generate the instance description:
- valid: whether the mask indicates a valid instancneglect backgrounds or abstract objects)
- nameL: name of the instance
- feature: list of features of the instance
- usage: list of usage of the instance

The generation of instance description follows a two-stage process:
- First pass the processed image to VLM to generate a detailed description of the instance
- Second pass the ouput to LLM to parse the detailed description and generated structured output (in a json format)

After description generation, the instance description is stored in the json file.

```
processing description for parent instance
Total image processing time: 0.0957 seconds
{'valid': True, 'name': 'Tube of cream or lotion', 'feature': ['The tube has a squeeze dispenser at the top.', "T
he tube is labeled with the brand 'Rituals'.", 'The tube appears to be partially used, as it is not completely fu
ll.'], 'usage': ['To contain and dispense a skincare or cosmetic product']}
processing description for parent instance
Total image processing time: 0.1051 seconds
{'valid': True, 'name': 'Cosmetic Bag', 'feature': ['Zipper closure', 'Handle for carrying', 'Rectangular shape']
, 'usage': ['To store and organize toiletries', 'To store personal care items']}
processing description for parent instance
Total image processing time: 0.0843 seconds
{'valid': False}
processing description for parent instance
Total image processing time: 0.0938 seconds
{'valid': False}
processing description for parent instance
Total image processing time: 0.0974 seconds
{'valid': True, 'name': 'Cosmetic Bag', 'feature': ['Zipper closure', 'Flat rectangular shape', 'Fabric material'
], 'usage': ['Store and organize toiletries', 'Store and organize personal care items']}
processing description for parent instance
Total image processing time: 0.0943 seconds
Rate limit exceeded. Retrying in 2.24 seconds... (Attempt 1/5)
```

####
