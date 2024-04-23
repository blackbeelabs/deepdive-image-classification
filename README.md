# Image Classification with Pytorch

## Introduction
Image Classification with Pytorch is a simple implementation of a deep learning project for NLP, in particular PyTorch. This project solves for image classification for MNIST


The workflow contains a couple of steps:
1. Ingestion to a local folder
2. Building a model
3. Evaluation of a model

## Quickstart
Initialise your Python environment. I use `pyenv` / `virtualenv`, and 3.10.1
```
pyenv virtualenv 3.10.1 imgclassification
pyenv activate imgclassification
pip install -r requirements.txt
```

Set the `PYTHONPATH` and go to the `src` folder
```
export PYTHONPATH="/path/to/deepdive-image-classification/src"
cd /path/to/deepdive-image-classification/src
```

Run `download_images.py` 
Run `train.py` 
Run `evaluate.py` 

Observe that the models are saved in:
```
├── assets
│   └── model
│   │   ├──model-baseline-20240308T104310.pkl
│   │   └── modelprofile-baseline-20240308T104310.json
```

Observe that the reports are in:
```
├── assets
│   └── model
│   │   └── report-baseline-20240308T104310.json
```
