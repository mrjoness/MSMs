BATCH --mail-type=ALL
#!/bin/sh

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

module add jupyter

launch_jupyter_notebook
