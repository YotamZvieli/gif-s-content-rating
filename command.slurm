#! /bin/sh

#SBATCH --job-name=pre_train_model_inference
#SBATCH --output=home/ai_center/ai_users/guy/.yotamZ/pre_trained.out # redirect stdout
#SBATCH --error=home/ai_center/ai_users/guy/.yotamZ/pre_trained.err # redirect stderr
#SBATCH --partition=studentkillable
#SBATCH --time=1
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1 # GPUs in total
#SBATCH --account=yotamzvieli


python awesome.py