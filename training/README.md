# Landmarks Detection Training

## File Structure
    .
    ├── app
    │   └── landmark
    │       │—— main.py 
    │       │—— runner.py 
    │       │—— config.yaml  ##Training configuration file
    │       └── __init__.py
    ├── data
    │   │—— train
    │   │   │—— images 
    │   │   └── labels
    │   └── test
    │       │—— images  
    │       └── labels
    ├── baseUtils
    ├── datasets
    ├── data_aug
    ├── loss
    ├── networks
    ├── optim
    └── README.md



## Usage
1. Prepare the training dataset
```
Detailed examples are available in the dataset directory for reference.
```

2. Activate the virtual environment
```
conda activate deploy
```

3. Training
```
python app/landmark/main.py
```