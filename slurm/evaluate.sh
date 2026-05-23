#!/bin/bash
#SBATCH --job-name=gnn-eval
#SBATCH --output=slurm_logs/eval_%j.out
#SBATCH --error=slurm_logs/eval_%j.err
#SBATCH --partition=TO_CHANGE           # <-- Change to your partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

# ============================================================================ #
#  SLURM Job — Evaluate all trained policy models
#
#  Usage:
#    mkdir -p slurm_logs
#    sbatch slurm/evaluate.sh
# ============================================================================ #

PROJECT_DIR="$HOME/GNN-DigiTwin-Wireless"    # <-- Change if needed

echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Date:         $(date)"
echo "============================================================"

cd "$PROJECT_DIR" || exit 1

# Activate env — uncomment one:
# source activate gnn_env
# conda activate gnn_env
# source venv/bin/activate

python -u evaluate_models.py \
    --checkpoint-dir checkpoints_v2 \
    --data-dir Data_cleaned \
    --output-dir evaluation_results

echo ""
echo "Evaluation finished at $(date)"
