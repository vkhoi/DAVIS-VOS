#!/bin/bash
#SBATCH --output scripts/output/train_afstream.out
#SBATCH --error scripts/output/train_afstream.out
#SBATCH --time=05:00:00
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --gres=gpu:1
#SBATCH --mem 16gb

srun python train_appearance_flow_net.py --batch_size=1 --resume_epoch=0 --n_epochs=10 --log=100 --cuda=1
