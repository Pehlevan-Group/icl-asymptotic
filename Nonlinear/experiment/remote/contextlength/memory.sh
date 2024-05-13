#!/bin/bash
# 
#SBATCH --job-name=test
#SBATCH -c 10
#SBATCH -t 3-00:00:00
#SBATCH -p kempner
#SBATCH --gres=gpu:1
#SBATCH --mem=320000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/contextlength/outputdump/run_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/ICLexperiments/experiment/remote/contextlength/outputdump/run_%A.err
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01
source activate try4
export XLA_PYTHON_CLIENT_PREALLOCATE=false

parentdir="resultsdump"
newdir="$parentdir/${SLURM_JOB_NAME}"
mkdir "$newdir"
python memory.py $newdir