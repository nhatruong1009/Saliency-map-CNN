#!/bin/bash
#SBATCH --job-name=complex01     # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=08:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=end          # send mail when job ends
#SBATCH --mail-type=fail         # send mail if job fails
#SBATCH --partition=batch
#SBATCH --qos=short
#SBATCH --output=/media/lhbac04/code/CAMProjectXAI/%j-%x.out
#SBATCH --error=/home/lhbac04/code/CAMProjectXAI/%j-%x.err

# Load cuda
spack load cuda@11.7.0

# Config conda
eval "$(conda shell.bash hook)"
conda activate lnduc

# Change dir
cd /media/lhbac04/code/CAMProjectXAI

chmod +x ./run.sh

./run.sh