#!/bin/bash
# 
#SBATCH --job-name=4kap30
#SBATCH -c 10
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/fig67/outputdump/4kap30_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/fig67/outputdump/4kap30_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

parentdir="resultsdump"
newdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
errdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/errors"
pkldir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/pickles"
mkdir "$newdir"
mkdir "$errdir"
mkdir "$pkldir"
python kappasweep.py 10 $newdir 0 0 