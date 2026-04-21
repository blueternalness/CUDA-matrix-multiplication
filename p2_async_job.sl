#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p2_async_job.out
#SBATCH --gres=gpu:v100:1

module purge
module load legacy/CentOS7
module load gcc/8.3.0
module load nvidia-hpc-sdk/21.7
nvcc p2_async.cu -o p2_async
./p2_async
