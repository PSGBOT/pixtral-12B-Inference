# pixtral-12B-Inference
This repository contains Python code examples to interact with the PixTral 12B API.

## File Descriptions:
`generate.py`: simple script for generate object/part description from image using pixtral api.

## Setup
- Setup `your key` as environment variable `PIXTRAL_KEY`
- Download the sample dataset from feishu group `resources/Data/part_seg_dataset sample.zip` and extract it to a directory (DON'T put the dataset in this project directory)
- Specify the directory in `config/custom_cfg`

## Image processing
Process the source image and mask (area of interest) to guide the VLM for better understanding
- [ ] Crop the instance of interest
- [x] highlight the area of instance/part of interest
- [x] highlight the contour of the instance/part of interest

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

The generation of instance description follows a **two-stage inference** process:
- First pass the processed image to VLM to generate a detailed description of the instance
- Second pass the ouput to LLM to parse the detailed description and generated structured output (in a json format)

After description generation, the instance description is stored in the json file.

```
processing description for parent instance
Total image processing time: 0.0957 seconds
{'valid': True, 'name': 'Tube of cream or lotion', 'feature': ['The tube has a squeeze dispenser at the top.', "The tube is labeled with the brand 'Rituals'.", 'The tube appears to be partially used, as it is not completely full.'], 'usage': ['To contain and dispense a skincare or cosmetic product']}
processing description for parent instance
Total image processing time: 0.1051 seconds
{'valid': True, 'name': 'Cosmetic Bag', 'feature': ['Zipper closure', 'Handle for carrying', 'Rectangular shape'], 'usage': ['To store and organize toiletries', 'To store personal care items']}
processing description for parent instance
Total image processing time: 0.0843 seconds
{'valid': False}
processing description for parent instance
Total image processing time: 0.0938 seconds
{'valid': False}
processing description for parent instance
Total image processing time: 0.0974 seconds
{'valid': True, 'name': 'Cosmetic Bag', 'feature': ['Zipper closure', 'Flat rectangular shape', 'Fabric material'], 'usage': ['Store and organize toiletries', 'Store and organize personal care items']}
processing description for parent instance
Total image processing time: 0.0943 seconds
Rate limit exceeded. Retrying in 2.24 seconds... (Attempt 1/5)
...
```

#### Relation Description
1. For the child masks under an instance, generate parent-child key-value pairs for relation generation. Each pair correspond to one KAF.

Take `id 1/mask1` as an example:
```bash
id 1/
├── mask1.png
└── mask1/
    ├── mask0/
    │   ├── mask0.png
    │   ├── mask1.png
    │   └── mask2.png
    ├── mask0.png
    ├── mask1/
    │   ├── mask0.png
    │   ├── mask1.png
    │   └── mask2.png
    └── mask1.png
```
Generate 3 keys(parent) for `id 1/mask1`: ['id 1/mask1/mask1.png', 'id 1/mask1/mask0.png', 'id 1/mask1.png']
Each key corresponds to a list of child mask pairs:
- `id 1/mask1/mask1.png` : 3 pairs
- `id 1/mask1/mask0.png` : 3 pairs
- `id 1/mask1.png` : 1 pairs

2. For each pair, process the image using part-level masks

![Pasted image (21)](https://github.com/user-attachments/assets/1b56eddc-d751-4883-84f0-2308ef5193e3)

3. Pass the processed image to model, following similar **two-stage inference** process to generate structured output for part relation.

```
Dataset contains 3 images

existing description, skipping vlm

['/home/cyl/Reconst/Data/Sample dataset/part_seg_dataset/id 1/mask1/mask1.png', '/home/cyl/Reconst/Data/Sample dataset/part_seg_dataset/id 1/mask1/mask0.png', '/home/cyl/Reconst/Data/Sample dataset/part_seg_dataset/id 1/mask1.png']

Total image processing time: 0.2574 seconds
Total image processing time: 0.0639 seconds

{'response': '### Kinematic Analysis of Soap Dispenser

#### Image 0: Highlighted Part (Part 0)
- **Technical Name**: Pump mechanism

#### Image 1: Highlighted Part (Part 1)
- **Technical Name**: Dispensing nozzle

### Kinematic Relationship Analysis

#### Joint Type and Connection
1. **Joint Type**: Revolute
   - **Description**: The pump mechanism and thedispensing nozzle are connected in such a way that the nozzle can rotate around a single axis relative to the pump mechanism.
   - **Axis of Movement**: Radial
   - **Type of Connection**: Controlled
   - **Functional Purpose**: Allows the user to direct the flow of soap by rotating the nozzle.
   - **Kinematic Root**: "0" (The pump mechanism remains stationary while the nozzle rotates.)

### Summary
- **Highlighted Part in Image 0**: Pump mechanism
- **Highlighted Part in Image 1**: Dispensing nozzle
- **Joint Types**:
  1. Revolute joint with radial movement, controlled connection, functional for directing soap flow, kinematic root is the pump mechanism.
  2. Fixed joint, static connection, ensures effective operation of the pump mechanism, kinematic root is the bottle.

### Conclusion
The kinematic analysis reveals that the pump mechanism is the primary stable part, with the dispensing nozzle capable of radial rotation to direct the soap flow. The fixed connection of the pump mechanism to the bottle ensures its proper functioning within the soap dispenser.'}


{'part1_name': 'Pump mechanism', 'part2_name': 'Dispensing nozzle', 'kinematic_joints': [{'joint_type': 'revolute', 'joint_movement_axis': 'radial', 'is_static': 'false', 'purpose': 'Allows the user to direct the flow of soapby rotating the nozzle.'}], 'root_part_id': '0'}

```

4. *TODO*
Extract information from structured outputs. Combine the results under same key together to get the raw data for KAF.
