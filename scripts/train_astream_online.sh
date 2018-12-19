#!/bin/bash
#SBATCH --output scripts/output/train_astream_online.out
#SBATCH --error scripts/output/train_astream_online.out
#SBATCH --time=05:00:00
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --gres=gpu:1
#SBATCH --mem 16gb

srun python train_appearance_net_online.py --net=VOSModel_epoch-20 --fname=blackswan --n_iterations=200 --log=5 --cuda=1
