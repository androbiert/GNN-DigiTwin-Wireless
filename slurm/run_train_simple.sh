#!/bin/bash
#SBATCH --job-name=gnn-v2
#SBATCH --output=slurm_logs/train_v2_%j.out
#SBATCH --error=slurm_logs/train_v2_%j.err
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# ============================================================================ #
# Exécution simple avec SLURM (Une seule commande, un seul GPU)
# ============================================================================ #

echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Node:         $(hostname)"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Date:         $(date)"
echo "============================================================"

# Aller dans le dossier du projet
cd "$HOME/GNN-DigiTwin-Wireless" || exit 1

# Lancer exactement la commande que tu as demandée
python train_scenarios.py --data-dir Data_cleaned --scenario SC01 --target delay --split-by-policy --resume --epochs 100 --model v2 --checkpoint-dir checkpoints_v2

echo ""
echo "Entraînement terminé à $(date)"
