#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH -o IO/output.%j
#SBATCH -e IO/error.%j
#SBATCH --array=0-1000
#SBATCH --partition=packable
#SBATCH --account=j1042
#SBATCH --job-name=Atm
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3G

#Need to delay job start times by random number to prevent overloading system:
sleep `expr $RANDOM % 60`

#The MEGAlib environment first needs to be sourced:
source your/path/COSI.sh

#Change to home directory and run job
cd $SLURM_SUBMIT_DIR
mkdir sim_$SLURM_ARRAY_TASK_ID
scp atmosphere.geo "Atmosphere_Isotropic.source" run_atm_sims.py sim_$SLURM_ARRAY_TASK_ID
cd sim_$SLURM_ARRAY_TASK_ID
python run_atm_sims.py
