#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=kp124
#SBATCH -C c16
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
lscpu
echo "mv2 trial1"
./mv2 | tee -a kingspeak_mv2.$SLURM_JOB_ID\.log
echo "mv2 trial 2"
./mv2 | tee -a kingspeak_mv2.$SLURM_JOB_ID\.log
echo "mv2 trial 3"
./mv2 | tee -a kingspeak_mv2.$SLURM_JOB_ID\.log
