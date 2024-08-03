#!/bin/bash
# 1alpha_NONLIN.sbatch
# 
#SBATCH --job-name=1alpha_NONLIN
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/1alpha_NONLIN_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/1alpha_NONLIN_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python finitebayesrun.py 1