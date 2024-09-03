#!/bin/bash
# kappa10alphasweep_lownoise.sbatch
# 
#SBATCH --job-name=kappa10alphasweep_lownoise
#SBATCH -c 1
#SBATCH -t 1-00:00:00
#SBATCH -p seas_compute
#SBATCH --mem=48000
#SBATCH -o /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/kappa10alphasweep_lownoise%A.out
#SBATCH -e /n/holyscratch01/pehlevan_lab/Lab/mletey/icl-asymptotic/Linear/Fig3_Mem_ICL_Transition_Task_Diversity/bayedump/kappa10alphasweep_lownoise%A.err
#SBATCH --array=1-500%50
#SBATCH --mail-type=END
#SBATCH --mail-user=maryletey@fas.harvard.edu

source activate try4

parentdir="bayedump"
newdir="$parentdir/job_${SLURM_JOB_NAME}"
mkdir "$newdir"
python finitebayesrun.py $newdir 10 $SLURM_ARRAY_TASK_ID