#!/bin/bash

#SBATCH --gpus=8
#SBATCH --gpus-per-node=8
#SBATCH --constraint=v100
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=40
#SBATCH --time=04:00:00
#SBATCH --mem=100G

module unuse /ibex/scratch/shaima0d/software/modulefiles
module load machine_learning/2022.11

port=$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')
node=$(hostname -s) 
echo -e "
ssh -L ${port}:${node}:${port} ${USER}@glogin.ibex.kaust.edu.sa 
"
nvdashboard  ${port} &
export OMP_NUM_THREADS=1
#srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} torchrun --standalone --nnodes 1 --nproc_per_node ${SLURM_GPUS_PER_NODE}  imagenet_ddp.py --batch-size 256 --num-workers 8 --epochs 1 --print-interval 25

for i in 1 2 4 8
 do
	srun -n 1 -N 1 -c ${SLURM_CPUS_PER_TASK} torchrun --standalone --nnodes 1 --nproc_per_node ${i}  imagenet_ddp.py --batch-size 256 --num-workers 5 --epochs 1 --print-interval 25 &>  log_${i}.txt
done
