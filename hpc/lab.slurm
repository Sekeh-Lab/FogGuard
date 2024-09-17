#!/bin/bash
#SBATCH --job-name=prune  # Job name
#SBATCH --partition=precision               # Select partition
#SBATCH --cpus-per-task=4               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=1000:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request one GPU
#SBATCH --output=output/errors/tbst_%j.log   # Standard output and error log
#SBATCH --error=output/errors/error_%j
#SBATCH --mail-type=BEGIN,END,FAIL          # Mail events
#SBATCH --mail-user=soheil.gharatappeh@maine.edu


echo "====================================================="
pwd; hostname; date
echo "====================================================="

# module load gnu8/8.3.0 mvapich2/2.3.2  /21.08
# echo "Loading modules"
# module load singularity nv/pytorch anaconda3 
# echo "Modules loaded"

# . $CONDA_BIN/conda-init
# conda activate ib

python "$@"
echo "====================================================="
date

