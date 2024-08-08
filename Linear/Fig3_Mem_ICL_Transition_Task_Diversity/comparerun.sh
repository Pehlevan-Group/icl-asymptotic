#!/bin/bash
# 100alpha.sbatch
# 
#SBATCH --job-name=100alpha
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu
#SBATCH --gpus=1
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/100alpha_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/100alpha_%A.out
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python finitebayesrun.py 100