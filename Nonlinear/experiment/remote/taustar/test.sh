#!/bin/bash
# 1d60d_arrays.sbatch
# 
#SBATCH --job-name=1L60d
#SBATCH -c 10
#SBATCH -t 3-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/taustar/outputdump/1L60d_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/taustar/outputdump/1L60d_%A_%a.err
#SBATCH --array=1-16
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

parentdir="resultsdump"
newdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
pkldir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/pickles"
errdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/errors"
mkdir "$newdir"
mkdir "$pkldir"
mkdir "$errdir"
python run.py $newdir 60 $SLURM_ARRAY_TASK_ID