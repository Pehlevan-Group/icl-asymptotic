#!/bin/bash
# 1fig6_arrays.sbatch
# 
#SBATCH --job-name=1fig6
#SBATCH -c 10
#SBATCH -t 2-00:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/fig67/outputdump/run_%A_%a.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/fig67/outputdump/run_%A_%a.err
#SBATCH --array=1-80%20
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

calculate_indices() {
    tauind=$(( ($1 - 1) / 4 ))
    avgind=$(( ($1 - 1) % 4 ))
}
calculate_indices $SLURM_ARRAY_TASK_ID

parentdir="resultsdump"
newdir="$parentdir/fig6"
pkldir="$parentdir/fig6/pickles"
errdir="$parentdir/fig6/errors"
mkdir "$newdir"
mkdir "$pkldir"
mkdir "$errdir"
python run.py $newdir $tauind $avgind 0.1