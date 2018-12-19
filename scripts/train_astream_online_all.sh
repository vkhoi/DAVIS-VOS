#!/bin/bash
#SBATCH --output scripts/output/train_astream_online_all.out
#SBATCH --error scripts/output/train_astream_online_all.out
#SBATCH --time=05:00:00
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --gres=gpu:1
#SBATCH --mem 16gb

srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=blackswan --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=bmx-trees --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=breakdance --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=camel --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=car-roundabout --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=car-shadow --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=cows --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=dance-twirl --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=dog --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=drift-chicane --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=drift-straight --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=goat --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=horsejump-high --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=kite-surf --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=libby --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=motocross-jump --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=paragliding-launch --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=parkour --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=scooter-black --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_net_online.py --net=VOSModel_epoch-50 --fname=soapbox --n_iterations=$1 --log=50 --cuda=1