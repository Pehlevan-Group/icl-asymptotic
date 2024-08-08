#!/bin/bash
# 100alpha.sbatch
# 
#SBATCH --job-name=100alpha
#SBATCH -c 1
#SBATCH -t 08:00:00
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/100alpha_%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/100alpha_%A.err
#SBATCH --array=1-40
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4
parentdir="bayedump"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py $newdir 100 $SLURM_ARRAY_TASK_ID