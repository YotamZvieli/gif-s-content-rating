#! /bin/sh

#SBATCH --job-name=pre_train_model_inference
#SBATCH --output=~/pre_trained.out # redirect stdout
#SBATCH --error=~/pre_trained.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=200 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=70000 # CPU memory (MB)
#SBATCH --cpus-per-task=5 # CPU cores per process
#SBATCH --gpus=5 # GPUs in total

python ~/gif-s-content-rating/Test-pre-trained-model.py