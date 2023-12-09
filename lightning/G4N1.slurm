#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --gpus-per-node=4
#SBATCH --constraint=a100
#SBATCH --time=03:00:00
#SBATCH --reservation=DS-TRAINING


#source /ibex/ai/home/$USER/miniconda3/bin/activate dist-pytorch
module load dl
module load pytorch

export NCCL_DEBUG=INFO
#export NCCL_SOCKET_IFNAME=ib0
export PYTHONFAULTHANDLER=1

export DATA_DIR=/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg
#export DATA_DIR=/ibex/ai/reference/CV/tinyimagenet


export NV_PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
node=$(hostname -s)

#nvdashboard ${NV_PORT} &
#echo "
#ssh -L localhost:${NV_PORT}:${node}.ibex.kaust.edu.sa:${NV_PORT} ${user}@glogin.ibex.kaust.edu.sa 
#"

time -p srun python multi_gpu.py --epochs=2 \
                         --num-nodes=${SLURM_NNODES} \
                         --gpus-per-node=${SLURM_GPUS_PER_NODE} \
                         --num-workers=10  \
                         --precision=32
