#!/bin/bash
#SBATCH --output scripts/output/train_afstream_online%j.out
#SBATCH --error scripts/output/train_afstream_online%j.out
#SBATCH --time=05:00:00
#SBATCH --partition=dpart
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --mem 16gb

srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-5 --fname=$1 --n_iterations=$2 --log=20 --cuda=1