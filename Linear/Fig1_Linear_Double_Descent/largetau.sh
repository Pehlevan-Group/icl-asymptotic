#!/bin/bash
# pnasA1_reg01.sbatch
# 
#SBATCH --job-name=pnasA100
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig1_Linear_Double_Descent/outputs/pnasA1_reg01_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig1_Linear_Double_Descent/outputs/pnasA1_reg01_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python largetau.py 100 1