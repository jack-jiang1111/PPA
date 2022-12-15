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
echo "MMTU trial1"
./mmtu | tee -a kingspeak_mmtu.$SLURM_JOB_ID\.log
echo "MMTU trial 2"
./mmtu | tee -a kingspeak_mmtu.$SLURM_JOB_ID\.log
echo "MMTU trial 3"
./mmtu | tee -a kingspeak_mmtu.$SLURM_JOB_ID\.log
