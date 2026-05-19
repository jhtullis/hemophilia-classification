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

# Note: Phase 1 (5 epochs, head-only warmup) completes in the first job.
# Phase 2 (full fine-tune) continues in subsequent jobs.
# The checkpoint stores 'phase' so resumption is transparent.

module load miniforge3
conda activate fibrin

export WANDB_MODE=offline

python train.py --model-type 3class_finetune \
    --max-epochs 10000 --epochs-per-job 200 --resume

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch slurm/train_3class_finetune.sh
fi
