#!/bin/bash 

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --gpus-per-node=1
##SBATCH --constraint=a100
#SBATCH --cpus-per-task=20
##SBATCH --mem=400G
#SBATCH --account=ibex-cs
##SBATCH --reservation=A100

 
#ad environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# setup the environment
module unuse /ibex/scratch/shaima0d/software/modulefiles

# You can use the machine learning module 
module load machine_learning/2022.11
# or you can activate the conda environment directly by uncommenting the following lines
#export ENV_PREFIX=$PWD/env
#conda activate $ENV_PREFIX

#export DATA_DIR=/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
export DATA_DIR=/ibex/ai/home/shaima0d/tiny-imagenet-200
# setup ssh tunneling
# get tunneling info 
export XDG_RUNTIME_DIR=/tmp node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
tb_port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
echo ${node} pinned to port ${port} on ${submit_host} 

# print tunneling instructions  
echo -e " 
${node} pinned to port ${port} on ${submit_host} 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 
 
ssh -L ${port}:${node}.ibex.kaust.edu.sa:${port} -L ${tb_port}:${node}:${tb_port} ${user}@glogin.ibex.kaust.edu.sa 
 
Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop.
" >&2 
 
# Run Jupyter 
#jupyter notebook --no-browser --port=${port} --port-retries=0 --ip=${node}

#tensorboard --logdir $PWD/logs --host ${node} --port ${tb_port} &

# launch jupyter server
#jupyter ${1:-lab} --no-browser --port=${port} --port-retries=0  --ip=${node}.ibex.kaust.edu.sa
jupyter-lab --no-browser --port=${port} --port-retries=0  --ip=${node}.ibex.kaust.edu.sa
