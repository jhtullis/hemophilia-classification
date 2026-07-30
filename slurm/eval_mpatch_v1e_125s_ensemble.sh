#!/bin/bash --login
#SBATCH --job-name=fibrin_eval_mpatch_v1e_125s_ensemble
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/fibrin_eval_mpatch_v1e_125s_ensemble_%j.out
#SBATCH --error=slurm/logs/fibrin_eval_mpatch_v1e_125s_ensemble_%j.err
#SBATCH --mail-type=FAIL,END

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

module load miniforge3
conda activate fibrin

python evaluate_holdout_125s_ensemble.py
