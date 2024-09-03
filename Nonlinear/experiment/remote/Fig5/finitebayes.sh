#!/bin/bash
# 0p5alpha.sbatch
# 
#SBATCH --job-name=0p5alpha
#SBATCH -c 1
#SBATCH -t 08:00:00
#SBATCH -p kempner
#SBATCH --gpus=1
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/Fig5/bayedump/0p5alpha%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Nonlinear/experiment/remote/Fig5/bayedump/0p5alpha%A.err
#SBATCH --array=1-20%12
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu
#SBATCH --account=kempner_pehlevan_lab

source activate try4
parentdir="bayedump"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py $newdir 0.5 $SLURM_ARRAY_TASK_ID