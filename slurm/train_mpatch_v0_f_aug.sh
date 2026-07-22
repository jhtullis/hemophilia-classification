#!/bin/bash --login
#SBATCH --job-name=fibrin_mpatch_v0_f_aug
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --partition=cs,cs2,cssp1,dw,dwmatrix,eng,m13h,msg
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_mpatch_v0_f_aug_%j.out
#SBATCH --error=slurm/logs/fibrin_mpatch_v0_f_aug_%j.err
#SBATCH --mail-type=FAIL
#SBATCH --signal=B:USR1@300

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

trap 'echo "Wall time approaching — resubmitting..."; sbatch "${PROJECT_DIR}/slurm/train_mpatch_v0_f_aug.sh"; exit 0' USR1

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline
export WANDB_DIR="${PROJECT_DIR}/models/mpatch_v0_f_aug"

python train.py --model-type mpatch_v0_f_aug \
    --max-epochs 10000 --epochs-per-job 600 --resume --preload &
PY_PID=$!
wait $PY_PID

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch "${PROJECT_DIR}/slurm/train_mpatch_v0_f_aug.sh"
fi
