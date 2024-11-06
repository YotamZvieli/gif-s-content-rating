#! /bin/sh

#SBATCH --job-name=fully_mae
#SBATCH --output=/home/ai_center/ai_data/ruth_test/.hanan/fully_mae.out
#SBATCH --error=/home/ai_center/ai_data/ruth_test/.hanan/fully_mae.err
#SBATCH --partition=studentkillable
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=70000 # CPU memory (MB)
#SBATCH --cpus-per-task=5 # CPU cores per process
#SBATCH --gpus=5 # GPUs in total

python ~/gif-s-content-rating/hanan/mae.py
