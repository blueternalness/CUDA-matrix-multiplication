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
module load nvidia-hpc-sdk

# Compile the Parallel Issue Loops program
nvcc p2_async.cu -o p2_async

# Run Problem 2.2
echo "=========================================="
echo " Problem 2.2: Parallel Issue Loops"
echo "=========================================="
./p2_async