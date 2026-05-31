#!/bin/bash --login
#SBATCH --job-name=fibrin_patch_v1b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_patch_v1b_%j.out
#SBATCH --error=slurm/logs/fibrin_patch_v1b_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "${PROJECT_DIR}/slurm/train_patch_v1b.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/models/patch_5class_v1b"

python train.py --model-type patch_v1b \
    --max-epochs 10000 --epochs-per-job 600 --resume --preload &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "${PROJECT_DIR}/slurm/train_patch_v1b.sh"
fi
