#! /bin/sh

#SBATCH --job-name=pre_train_model_inference
#SBATCH --output=home/ai_center/ai_users/guy/.yotamZ/pre_trained.out # redirect stdout
#SBATCH --error=home/ai_center/ai_users/guy/.yotamZ/pre_trained.err # redirect stderr
#SBATCH --partition=studentkillable

python awesome.py