#!/bin/bash
# 40increase.sbatch
# 
#SBATCH --job-name=40increase
#SBATCH -c 10
#SBATCH -t 1-00:00:00
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/Fig5/outputdump/40increase%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/Fig5/outputdump/40increase%A_%a.err
#SBATCH --array=1-260%30
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    tauind=$(( ($1 - 1) / 10 ))
    avgind=$(( ($1 - 1) % 10 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="resultsdump"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
pkldir="$parentdir/job_${SLURM_JOB_NAME}/pickles"
mkdir "$newdir"
mkdir "$pkldir"
python run.py $newdir 40 $tauind $avgind