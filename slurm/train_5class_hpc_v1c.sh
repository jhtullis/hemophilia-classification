#!/bin/bash --login
#SBATCH --job-name=fibrin_5class_hpc_v1c
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_5class_hpc_v1c_%j.out
#SBATCH --error=slurm/logs/fibrin_5class_hpc_v1c_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "${PROJECT_DIR}/slurm/train_5class_hpc_v1c.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/models/5class_hpc_v1c"

python train.py --model-type 5class_hpc_v1c \
    --max-epochs 10000 --epochs-per-job 1500 --resume --preload &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "${PROJECT_DIR}/slurm/train_5class_hpc_v1c.sh"
fi
