#!/bin/bash --login
#SBATCH --job-name=fibrin_mpatch_v1e_125s_flat
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --partition=eng,m13h,mgh
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_mpatch_v1e_125s_flat_%j.out
#SBATCH --error=slurm/logs/fibrin_mpatch_v1e_125s_flat_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "${PROJECT_DIR}/slurm/train_mpatch_v1e_125s_flat.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/models/mpatch_v1e_125s_flat"

python train.py --model-type mpatch_v1e_125s_flat \
    --max-epochs 10000 --epochs-per-job 100 --resume &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "${PROJECT_DIR}/slurm/train_mpatch_v1e_125s_flat.sh"
fi
