#! /bin/sh

#SBATCH --job-name=pre_train_model_inference
#SBATCH --output=/home/ai_center/ai_users/guy/.yotamZ/pre_trained.out # redirect stdout
#SBATCH --error=/home/ai_center/ai_users/guy/.yotamZ/pre_trained.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=1 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=1 # GPUs in total

python ~/gif-s-content-rating/Test-pre-trained-model.py