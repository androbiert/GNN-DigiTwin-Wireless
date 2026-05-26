#!/bin/bash
#SBATCH --job-name=gnn_queue_delay
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --partition=main

#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

#SBATCH --time=24:00:00

#SBATCH --array=0-3%2

cd ~/GNN-DigiTwin-Wireless

source venv/bin/activate

mkdir -p logs

QUEUES=("50KiB" "100KiB" "2MiB" "10MiB")
QUEUE=${QUEUES[$SLURM_ARRAY_TASK_ID]}

echo "======================================"
echo "Queue: $QUEUE"
echo "Job ID: $SLURM_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "======================================"

python train_scenarios_by_queue.py \
  --target delay \
  --queue-size "$QUEUE" \
  --epochs 100 \
  --checkpoint-dir checkpoints_queue \
  --data-dir data_cleaned \
  --device cuda