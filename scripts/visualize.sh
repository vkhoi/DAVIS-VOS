#!/bin/bash
#SBATCH --output scripts/output/visualize.out
#SBATCH --error scripts/output/visualize.out
#SBATCH --time=05:00:00
#SBATCH --partition=class
#SBATCH --qos=class
#SBATCH --mem 16gb

srun python visualize.py --fname $1
