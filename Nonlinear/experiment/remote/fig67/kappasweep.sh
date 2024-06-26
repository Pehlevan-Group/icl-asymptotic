#!/bin/bash
# kap30a1_arrays.sbatch
# 
#SBATCH --job-name=kap30a1
#SBATCH -c 10
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/fig67/outputdump/kap30a1_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/fig67/outputdump/kap30a1_%A_%a.err
#SBATCH --array=1-110%30
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    kappaind=$(( ($1 - 1) / 5 ))
    avgind=$(( ($1 - 1) % 5 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="resultsdump"
newdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}"
errdir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/errors"
pkldir="$parentdir/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}/pickles"
mkdir "$newdir"
mkdir "$errdir"
mkdir "$pkldir"
python kappasweep.py 30 $newdir $kappaind $avgind