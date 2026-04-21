#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --partition=gpu 
#SBATCH --output=p2_job.out
#SBATCH --gres=gpu:v100:1

module purge
module load nvidia-hpc-sdk

# Compile the Shared Memory CUDA program
nvcc p2_memory.cu -o p2_memory

# Run Problem 1.2
echo "--- Running Problem 1.2: Shared Memory Tiling ---"
./p2_memory
