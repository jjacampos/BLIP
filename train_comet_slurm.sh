#!/bin/bash
## SLURM scripts have a specific format. 

### Section1: SBATCH directives to specify job configuration

## job name
#SBATCH --job-name=fine_tune_blip

## filename for job standard output (stdout)
## %j is the job id, %u is the user id
#SBATCH --output=/fsx/jacampos/experiments/comet/train.out

## filename for job standard error output (stderr)
#SBATCH --error=/fsx/jacampos/experiments/comet/train.err

## partition name
#SBATCH --partition=a100

## number of gpus
#SBATCH --gpus-per-node=8

## number of tasks per node
#SBATCH --ntasks-per-node=8

### Section 2: Setting environment variables for the job
### Remember that all the module command does is set environment
### variables for the software you need to. Here I am assuming I
### going to run something with python.
### You can also set additional environment variables here and
### SLURM will capture all of them for each task
# Start clean
module purge

# Load what we need
source /data/home/jacampos/miniconda/etc/profile.d/conda.sh
conda activate blip

export TORCH_MODEL_ZOO=/fsx/jacampos/experiments/model_zoo/
export TORCH_HOME=/fsx/jacampos/experiments/model_zoo/

nvidia-smi

torchrun --nproc_per_node=8 train_comet.py --config ./configs/comet.yaml
