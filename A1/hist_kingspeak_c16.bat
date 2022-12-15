#!/bin/bash -x
#SBATCH -M kingspeak 
#SBATCH --account=owner-guest 
#SBATCH --partition=kingspeak-guest
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH -C c16
#SBATCH -c 16
#SBATCH --exclusive
#SBATCH -t 0:05:00
lscpu
echo "2 process"
mpirun -np 2 ./mat-vec-mul | tee -a P2.$SLURM_JOB_ID\.log
echo "4 process"
mpirun -np 4 ./mat-vec-mul | tee -a P2.$SLURM_JOB_ID\.log
echo "8 process"
mpirun -np 8 ./mat-vec-mul | tee -a P2.$SLURM_JOB_ID\.log
echo "16 process"
mpirun -np 16 ./mat-vec-mul | tee -a P2.$SLURM_JOB_ID\.log

