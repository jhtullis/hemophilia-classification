#!/bin/bash --login
#SBATCH --job-name=fibrin_5class_baseline
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_5class_baseline_%j.out
#SBATCH --error=slurm/logs/fibrin_5class_baseline_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "$SCRIPT_DIR/train_5class.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${SCRIPT_DIR}/../models/5class_hpc_baseline"

python train.py --model-type 5class_hpc_baseline \
    --max-epochs 10000 --epochs-per-job 100 --resume &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "$SCRIPT_DIR/train_5class.sh"
fi
