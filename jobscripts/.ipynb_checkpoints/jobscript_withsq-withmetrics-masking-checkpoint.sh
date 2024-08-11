#!/bin/bash


# Load the required module
module load python/3.10-anaconda

# Change to the SPADE directory
cd ../

pip install --proxy http://proxy:80 torch
pip install --proxy http://proxy:80 torchvision
pip install --proxy http://proxy:80 dominate
pip install --proxy http://proxy:80 dill
pip install --proxy http://proxy:80 scikit-image
pip install --proxy http://proxy:80 pytorch-fid


# Run the Python training script
python -u trainwithmetrics.py --name DRV47_withSQ_withmetrics_withmasking --dataset_mode pix2pix --dataroot datasets/DRV47/Withsq/train/ --no_instance --label_nc 0 --batchSize 1
