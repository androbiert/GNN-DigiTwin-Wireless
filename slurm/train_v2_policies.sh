#!/bin/bash
#SBATCH --job-name=gnn-v2-policy
#SBATCH --output=slurm_logs/train_v2_%A_%a.out
#SBATCH --error=slurm_logs/train_v2_%A_%a.err
#SBATCH --partition=TO_CHANGE           # <-- Change to your partition name
#SBATCH --gres=gpu:1                    # 1 GPU per policy model
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3                     # 4 policies: PF, MAXCI, DRR, MAXCI_MB

# ============================================================================ #
#  SLURM Array Job — Train one GNN model per scheduling policy (v2)
#
#  Each array task trains a different policy on its own GPU.
#  All 4 policies run in parallel.
#
#  Usage:
#    mkdir -p slurm_logs
#    sbatch slurm/train_v2_policies.sh
#
#  Monitor:
#    squeue -u $USER
#    tail -f slurm_logs/train_v2_<JOB_ID>_<TASK_ID>.out
# ============================================================================ #

# ── Configuration ────────────────────────────────────────────────────────── #
PROJECT_DIR="$HOME/GNN-DigiTwin-Wireless"    # <-- Change if needed
DATA_DIR="Data_cleaned"
SCENARIO="SC01"
TARGET="delay"
MODEL="v2"
EPOCHS=100
CHECKPOINT_DIR="checkpoints_v2"
SUBSAMPLE=1.0     # Use 100% of data (set to 0.2 for quick test)

# ── Map SLURM array index to policy ──────────────────────────────────────── #
POLICIES=("PF" "MAXCI" "DRR" "MAXCI_MB")
POLICY=${POLICIES[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (task $SLURM_ARRAY_TASK_ID)"
echo "  Policy:       $POLICY"
echo "  Model:        $MODEL"
echo "  Node:         $(hostname)"
echo "  GPU:          $CUDA_VISIBLE_DEVICES"
echo "  Date:         $(date)"
echo "============================================================"

# ── Environment ──────────────────────────────────────────────────────────── #
cd "$PROJECT_DIR" || exit 1

# Activate conda/venv if needed — uncomment the appropriate line:
# source activate gnn_env
# conda activate gnn_env
# source venv/bin/activate

# Verify GPU is visible
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Filter data paths for this policy ────────────────────────────────────── #
# Instead of using --split-by-policy (which trains ALL policies sequentially),
# we use --scenario to target SC01 and filter by policy via a small wrapper.
# The trick: create a temporary Python snippet that filters and runs.

python -u -c "
import sys, os
sys.path.insert(0, '.')

from wireless_gnn.scenario_registry import discover_scenarios, group_by_scenario, filter_for_target
from train_scenarios import train_scenario
from wireless_gnn.model import WirelessNetFermi

# Discover
root = os.path.abspath('.')
configs = discover_scenarios(root, data_dir='${DATA_DIR}', validate=True, verbose=True)
groups = group_by_scenario(configs)

sc_cfgs = groups.get('${SCENARIO}', [])
valid = filter_for_target(sc_cfgs, '${TARGET}')
policy_cfgs = [c for c in valid if c.scheduler == '${POLICY}']

if not policy_cfgs:
    print(f'ERROR: No configs for policy ${POLICY}')
    sys.exit(1)

data_paths = [c.data_path for c in policy_cfgs]
print(f'Training ${SCENARIO}_${POLICY} with {len(data_paths)} data files')

result = train_scenario(
    scenario_id    = '${SCENARIO}_${POLICY}',
    target         = '${TARGET}',
    data_paths     = data_paths,
    project_root   = root,
    hidden_dim     = 64,
    num_heads      = 4,
    iterations     = 8,
    dropout        = 0.1,
    epochs         = ${EPOCHS},
    lr             = 1e-3,
    patience       = 15,
    device_str     = 'auto',
    checkpoint_dir = '${CHECKPOINT_DIR}',
    seed           = 42,
    subsample_ratio= ${SUBSAMPLE},
    model_class    = WirelessNetFermi,
    resume         = True,
)
print('Training complete!')
"

echo ""
echo "Job finished at $(date)"
