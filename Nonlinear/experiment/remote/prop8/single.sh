#!/bin/bash
# 
#SBATCH --job-name=10tests
#SBATCH -c 10
#SBATCH -t 3-00:00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/prop8/outputdump/run_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/prop8/outputdump/run_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

parentdir="resultsdump"
newdir="$parentdir/${SLURM_JOB_NAME}"
pkldir="$parentdir/${SLURM_JOB_NAME}/pickles"
errdir="$parentdir/${SLURM_JOB_NAME}/errors"
mkdir "$newdir"
mkdir "$pkldir"
mkdir "$errdir"
python runsingle.py $newdir 3 5