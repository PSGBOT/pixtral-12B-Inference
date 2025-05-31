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
