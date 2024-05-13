#!/bin/bash
#
#SBATCH --job-name=taucurvesmall
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu,kempner,pehlevan_gpu,gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Everyone/run_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Everyone/run_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python tausmall.py 1

