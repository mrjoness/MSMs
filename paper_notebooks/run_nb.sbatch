#!/bin/bash

#SBATCH --mail-type=ALL
#SBATCH --mail-user=mikejones@uchicago.edu

#SBATCH --job-name=20mem

#SBATCH --output=Lag-1_GC-core.out
#SBATCH --partition=gm4-pmext
#SBATCH --nodes=1 # SET NUM NODES 
#SBATCH --ntasks-per-node=1 # SETS NUM MPI RANKS
#SBATCH --cpus-per-task=1 # SET NUM THREADS (cores)
#SBATCH --mem=30GB # REQUEST MEMORY
##SBATCH --constraint=gold-6248
#SBATCH --qos=gm4-cpu

# ex. python run_SNV_to_pkl.py mdtraj-pwdr-AT-all-326T-395-1000-190.npy prefix batch_size lag_time n_slow_modes reps

python run_single_to_pkl.py mdtraj-pwdr-20KL_AT-all_293K-221-10000-190.npy full-221_sm-8_l-64_scaled 50000 64 8 1
#python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-core_310K-86-10000-190.npy trim-10k_sm-2_l-64_scaled 50000 64 2 1
#python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-end_300K-148-10000-190.npy e-100_l-100_scaled 50000 100 6 1
#python run_single_to_pkl.py mdtraj-pwdr-20KL_GC-mid_309K-103-10000-190.npy e-100_l-100_scaled 50000 100 6 1


#python run_single_to_pkl.py mdtraj-pwdr-AT-all-326T-395-1000-190.npy e-ions-unscaled 50000 10 8 1
#python run_single_to_pkl.py mdtraj-pwdr-GC-mid-342T-430-1000-190.npy e-ions-unscaled 50000 10 8 1
#python run_single_to_pkl.py mdtraj-pwdr-GC-core-343T-480-1000-190.npy e-ions-unscaled 50000 10 8 1
#python run_single_to_pkl.py mdtraj-pwdr-GC-end-334T-403-1000-190.npy e-ions-unscaled 50000 10 8 1
