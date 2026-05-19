#!/bin/bash --login
#SBATCH --job-name=fibrin_5class_hpc_v0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/fibrin_5class_hpc_v0_%j.out
#SBATCH --error=slurm/logs/fibrin_5class_hpc_v0_%j.err
#SBATCH --mail-type=FAIL

module load miniforge3
conda activate fibrin

python train.py --model-type 5class_hpc_v0 \
    --max-epochs 10000 --epochs-per-job 200 --resume

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "Resubmitting continuation job..."
    sbatch slurm/train_5class_hpc_v0.sh
fi
