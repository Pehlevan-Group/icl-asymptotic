#!/bin/bash
# t20_a1000.sbatch
# 
#SBATCH --job-name=t20_a1000
#SBATCH -t 08:00:00
#SBATCH -p kempner
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/t20_a1000_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/t20_a1000_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source activate try4
python largealpha.py 50 1000