#!/bin/bash --login
#SBATCH --job-name=fibrin_3class_finetune
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_3class_finetune_%j.out
#SBATCH --error=slurm/logs/fibrin_3class_finetune_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

# Note: Phase 1 (5 epochs, head-only warmup) completes in the first job.
# Phase 2 (full fine-tune) continues in subsequent jobs.
# The checkpoint stores 'phase' so resumption is transparent.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "$SCRIPT_DIR/train_3class_finetune.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${SCRIPT_DIR}/../models/3class_hemo_finetune"

python train.py --model-type 3class_finetune \
    --max-epochs 10000 --epochs-per-job 100 --resume --preload &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "$SCRIPT_DIR/train_3class_finetune.sh"
fi
