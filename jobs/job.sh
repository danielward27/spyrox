#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4GB
#SBATCH --account=math026082
#SBATCH --output=%x_%A_%a.out
#SBATCH --time=6:00:00
#SBATCH --array=0-50

LOSS_NAME=$1
NUM_ROUNDS=$2

# Activate the conda environment
sleep 5
source ~/miniforge3/bin/activate
sleep 5
conda activate softcvi_validation_env

# Run the task
python -m scripts.run_task \
  --seed=$SLURM_ARRAY_TASK_ID \
  --loss-name="$LOSS_NAME" \
  --num-rounds="$NUM_ROUNDS" \
  --simulation-budget=20000 \
  --guide-steps=200000 \
  --surrogate-max-epochs=300
