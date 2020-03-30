#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikejones@uchicago.edu

#SBATCH --job-name=1g-2c

#SBATCH --output=out.out
#SBATCH --partition=gm4-pmext

#SBATCH --nodes=1 # SET NUM NODES 
#SBATCH --gres=gpu:1 # SET NUM GPUS
#SBATCH --ntasks-per-node=1 # SETS NUM MPI RANKS (1 PER GPU)
#SBATCH --cpus-per-task=2 # SET NUM THREADS (cores)
#SBATCH --mem=30GB # REQUEST MEMORY
#SBATCH --qos=gm4

#python run_single_to_pkl.py mdtraj-pwdr-20KL_AT-all_293K-221-10000-190.npy e-100_l-100_scaled 50000 100 6 1
#python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-end_300K-148-10000-190.npy e-100_l-100_scaled 50000 100 6 1
#python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-mid_309K-103-10000-190.npy e-100_l-100_scaled 50000 100 6 1
python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-core_310K-86-10000-190.npy e-100_l-100_scaled 50000 100 6 1