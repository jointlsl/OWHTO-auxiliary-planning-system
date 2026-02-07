# Landmarks Detection Pipeline

## File Structure
    .
    ├── training   ## Model training
    ├── inference  ## Model inference
    ├── README.md  
    └── requirements.txt

## Enviroment Installation
```
1. Miniconda Installation
```
For Windows version, refer to the link：https://blog.csdn.net/baidu_41805096/article details/108501099
```

2. Create a Virtual Environment
```
conda create -n deploy python=3.8
```

3.  Install Dependencies (CUDA 11.3)
```
pip install --no-index --find-links=pip_packages -r requirements.txt
```
Reference link: https://blog.csdn.net/weixin_35757704/article/details/120595248
