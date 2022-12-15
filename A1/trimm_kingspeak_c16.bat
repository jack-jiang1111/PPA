#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -C c16
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
lscpu
echo "Trimm_kij trial 1"
./trimm_kij | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log
echo "Trimm_kij trial 2"
./trimm_kij | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log
echo "Trimm_kij trial 3"
./trimm_kij | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log

echo "Trimm_ijk trial 1"
./trimm_ijk | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log
echo "Trimm_ijk trial 2"
./trimm_ijk | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log
echo "Trimm_ijk trial 3"
./trimm_ijk | tee -a kingspeak_trimm.$SLURM_JOB_ID\.log
