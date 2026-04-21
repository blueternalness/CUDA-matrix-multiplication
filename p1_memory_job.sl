#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p1_job.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk

# Compile the Global Memory CUDA program
nvcc p1_memory.cu -o p1_memory

# Run Problem 1.1
echo "--- Running Problem 1.1: Global Memory ---"
./p1_memory
