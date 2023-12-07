#!/bin/bash 

#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
##SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --reservation=DS-TRAINING

 
#ad environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

# setup the environment


# or you can activate the conda environment directly by uncommenting the following lines
module load dl
module load pytorch

export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet
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
 
# Run Tensorboard
tensorboard --logdir $PWD/logs --host ${node} --port ${tb_port} &

# launch jupyter server
jupyter-lab --no-browser --port=${port} --port-retries=0  --ip=${node}.ibex.kaust.edu.sa
