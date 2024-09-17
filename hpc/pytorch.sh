#!/bin/bash

#SBATCH --job-name=Compute correlation  # Job name
#SBATCH --cpus-per-task=4               # Run on a 4 cores per node
#SBATCH --nodes=1                       # Run on a 1 node
#SBATCH --ntasks=1                      # One process (shared-memory)
#SBATCH --partition=grtx                # Select partition
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=100:00:00                # Time limit hrs:min:sec
#SBATCH --gres=gpu:1                    # Request one GPU

#SBATCH --output=../output/logs/troubleshooting_%j.log   # Standard output and error log
#SBATCH --error=../output/logs/error_%j.log
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=soheil.gharatappeh@maine.edu


echo "====================================================="
pwd; hostname; date
echo "====================================================="

# module load gnu8/8.3.0 mvapich2/2.3.2 
module load anaconda3 
. $CONDA_BIN/conda-init
conda activate ib

echo "Loading modules"

module load nv/pytorch

# time srun singularity run ~/pytorch-sing.simg python correlation.py train
# singularity run ~/pytorch-sing.simg python correlation.py correlation
singularity exec --nv ~/pytorch-sing.sif python $1 $2
# srun python ~/ib_dl/src/correlation.py train
echo "====================================================="
date

