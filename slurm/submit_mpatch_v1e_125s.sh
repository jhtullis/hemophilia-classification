#!/bin/bash
# Submit all three mpatch_v1e_125s LR-schedule variants.
# They are independent and can run concurrently if multiple H200 GPUs are available.
#
# Usage (from project root):
#   bash slurm/submit_mpatch_v1e_125s.sh
mkdir -p slurm/logs
sbatch slurm/train_mpatch_v1e_125s.sh
sbatch slurm/train_mpatch_v1e_125s_flat.sh
sbatch slurm/train_mpatch_v1e_125s_plateau.sh
