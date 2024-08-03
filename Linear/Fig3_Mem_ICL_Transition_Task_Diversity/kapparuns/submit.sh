#!/bin/bash
# a01_100_80.sbatch
# 
#SBATCH --job-name=a01_100_80
#SBATCH --gpus 1
#SBATCH -t 08:00:00
#SBATCH -p kempner
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/results/a01_100_80_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/kapparuns/results/a01_100_80_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source activate try4
python kappa.py 100 0.1 100