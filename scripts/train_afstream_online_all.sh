#!/bin/bash
#SBATCH --output scripts/output/train_afstream_online_all.out
#SBATCH --error scripts/output/train_afstream_online_all.out
#SBATCH --time=05:00:00
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --gres=gpu:1
#SBATCH --mem 16gb

srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=blackswan --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=bmx-trees --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=breakdance --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=camel --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=car-roundabout --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=car-shadow --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=cows --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=dance-twirl --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=dog --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=drift-chicane --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=drift-straight --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=goat --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=horsejump-high --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=kite-surf --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=libby --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=motocross-jump --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=paragliding-launch --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=parkour --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=scooter-black --n_iterations=$1 --log=50 --cuda=1
srun python train_appearance_flow_net_online.py --net=AppearanceFlowModel_epoch-20 --fname=soapbox --n_iterations=$1 --log=50 --cuda=1