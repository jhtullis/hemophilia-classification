#!/bin/bash --login
#SBATCH --job-name=fibrin_eval_mpatch_v1e_lite_lc_aggregate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm/logs/fibrin_eval_mpatch_v1e_lite_lc_aggregate_%j.out
#SBATCH --error=slurm/logs/fibrin_eval_mpatch_v1e_lite_lc_aggregate_%j.err
#SBATCH --mail-type=FAIL,END

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load miniforge3
conda activate fibrin

python aggregate_learning_curve.py
