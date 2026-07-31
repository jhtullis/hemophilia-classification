#!/bin/bash --login
#SBATCH --job-name=fibrin_eval_mpatch_v1e_lite_lc
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:20:00
#SBATCH --output=slurm/logs/fibrin_eval_mpatch_v1e_lite_lc_%j.out
#SBATCH --error=slurm/logs/fibrin_eval_mpatch_v1e_lite_lc_%j.err
#SBATCH --mail-type=FAIL,END

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load miniforge3
conda activate fibrin

python evaluate_holdout_lite_lc.py
