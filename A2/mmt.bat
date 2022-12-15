#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=kp036
#SBATCH -C c16
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
lscpu
echo "MMT trial1"
./mmt | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log
echo "MMT trial 2"
./mmt | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log
echo "MMT trial 3"
./mmt | tee -a kingspeak_mmt.$SLURM_JOB_ID\.log