#!/bin/bash
# kappas_a100_t100_d50.sbatch
# 
#SBATCH --job-name=kappas_a100_t100_d50
#SBATCH -t 1-00:00:00
#SBATCH -p gpu
#SBATCH --gpus 1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig2_Context_Length/dump/kappas_a100_t100_d50_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig2_Context_Length/dump/kappas_a100_t100_d50_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python largealpha.py 50 100