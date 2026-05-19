#!/bin/bash --login
#SBATCH --job-name=fibrin_3class_scratch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_3class_scratch_%j.out
#SBATCH --error=slurm/logs/fibrin_3class_scratch_%j.err
#SBATCH --mail-type=FAIL

module load miniforge3
conda activate fibrin

python train.py --model-type 3class_scratch \
    --max-epochs 10000 --epochs-per-job 200 --resume

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch slurm/train_3class_scratch.sh
fi
