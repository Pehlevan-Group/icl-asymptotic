#!/bin/bash
# a100big.sbatch
# 
#SBATCH --job-name=a100big
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/a100big_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/runs/outputs/a100big_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
python largetau.py 80 100