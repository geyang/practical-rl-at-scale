#!/bin/bash

#SBATCH --nodes=20
#SBATCH --time=00:10:00
#SBATCH --partition=shas-testing
#SBATCH --job-name=mnist-job
#SBATCH --output=mnist-job.%j.out

# Activate Conda Environment
source activate mnist

srun python mnist_classify.py --seed %j
