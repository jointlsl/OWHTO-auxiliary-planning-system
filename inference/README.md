# Landmarks Detection Inference

## File Structure
    .
    ├── model
    │   ├── left
    │   │   └── llimb.pt
    │   └── right
    │       └── llimb.pt
    ├── README.md
    ├── src
    │   ├──__init__.py
    │   ├── landmark_detect.py
    │   └── script
    |       ├── networks
    │       └── utils
    │           ├── __init__.py
    │           └── kit.py
    └── test_data
        ├── demo1.png
        ├── demo2.png
        └── result

## Usage
```
1. conda activate deploy
2. python landmark_detect.py -image_path -save_path -region 1|2 -id Patient ID -debug True|False
```

## Parameters
```
-image | Input image path
--save | Output result save 
-save  | Output result save directory path
-region| 1: Left limb; 2: Right limb
-id    | Patient ID
-debug | Enable debug mode (visualization)

Training Example: python landmark_detect.py /mydir/datasets mydir/output_path 1 012342 True
```
