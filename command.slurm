#! /bin/sh

#SBATCH --job-name=pre_train_model_inference
#SBATCH --output=home/ai_center/ai_users/guy/.yotamZ/pre_trained.out
#SBATCH --error=home/ai_center/ai_users/guy/.yotamZ/pre_trained.err

python awesome.py