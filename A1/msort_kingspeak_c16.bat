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
echo "Merge Sort trial 1"
./msort | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
echo "Merge Sort trial 2"
./msort | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
echo "Merge Sort trial 3"
./msort | tee -a kingspeak_msort.$SLURM_JOB_ID\.log
