# pixtral-12B-Inference
This repository contains Python code examples to interact with the PixTral 12B API.

## File Descriptions:
`generate.py`: simple script for generate object/part description from image using pixtral api.

## Setup
- Setup `your key` as environment variable `PIXTRAL_KEY`
- Download the sample dataset from feishu group `resources/Data/part_seg_dataset sample.zip` and extract it to a directory (DON'T put the dataset in this project directory)
- Specify the directory in `config/custom_cfg`

## Results
```bash
python generate.py --object_name "microwave oven" --mode "part"

Total image processing time: 0.3092 seconds
Generating description for part: microwave oven
Using model: pixtral-12b-2409
Processing image: /home/cyl/Reconst/Data/Sample dataset/part_seg_dataset_sample/id 2.png
Using mask: /home/cyl/Reconst/Data/Sample dataset/part_seg_dataset_sample/id 2/mask0/mask_0.png

Description of the masked part:
--------------------------------------------------
{"name": "Microwave Door", "purpose": "Contains the microwave's interior where food is placed for heating", "text
": ""}
--------------------------------------------------
```
![Pasted image](https://github.com/user-attachments/assets/8017aae8-5dd8-4b53-a236-1d427e099890)

## Relation Generation
```bash
python relation_generator.py --dataset_dir "/home/cyl/Reconst/Data/Sample dataset/part_seg_dataset/part_seg_dataset.json" --src_image_dir "/hom
e/cyl/Reconst/Data/Sample dataset/src_img"
```
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
