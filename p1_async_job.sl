#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p1_async_job.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk

# Compile the Sequential Issue Loop program
nvcc p1_async.cu -o p1_async

# Run Problem 2.1
echo "=========================================="
echo " Problem 2.1: Sequential Issue Loop"
echo "=========================================="
./p1_async