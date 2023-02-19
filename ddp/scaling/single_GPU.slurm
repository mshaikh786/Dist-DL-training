#!/bin/bash 

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=10
#SBATCH --account=ibex-cs
#SBATCH --reservation=A100
#SBATCH --output=1gpu-%j.out
#SBATCH --error=1gpu-%j.out

 
#ad environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# setup the environment

# You can use the machine learning module 
#module load machine_learning/2022.11

# or you can activate the conda environment directly by uncommenting the following lines
source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch

export DATA_DIR=/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
#export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet

time -p python single_GPU.py
