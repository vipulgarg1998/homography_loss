#!/bin/bash

#SBATCH --time=07:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2080:3
#SBATCH --job-name=homography_training
#SBATCH --output=homography_training.out
#SBATCH --error=homography_training.err

python main_depth.py datasets/ShopFacade --cuda --weights /mundus/vgarg872/Documents/machine-perception/homography/logs/ShopFacade/local_homography/weights/epoch_1090.pth

# !200 + 1360 + 1100 + 1090 + 
