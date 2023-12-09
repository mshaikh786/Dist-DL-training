#!/bin/bash 

#SBATCH --time=00:05:00
#SBATCH --ntasks=4
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --gpus=4
#SBATCH --gpus-per-node=2
#SBATCH --constraint=a100
#SBATCH --reservation=DS-TRAINING

scontrol show job ${SLURM_JOBID} 
module load dl
module load jax/23.10-sif

export IMAGE=$JAX_IMAGE

srun -u -n ${SLURM_NTASKS} -N ${SLURM_NNODES} singularity run --nv $IMAGE \
	python query_v1.py


