#!/bin/bash
# Submit all four mpatch_v0 training jobs.
# They are independent and can run concurrently if multiple GPUs are available.
#
# Usage (from project root):
#   bash slurm/submit_mpatch_v0.sh
mkdir -p slurm/logs
sbatch slurm/train_mpatch_v0_a.sh
sbatch slurm/train_mpatch_v0_b.sh
sbatch slurm/train_mpatch_v0_c.sh
sbatch slurm/train_mpatch_v0_d.sh
sbatch slurm/train_mpatch_v0_e.sh
sbatch slurm/train_mpatch_v0_f.sh
sbatch slurm/train_mpatch_v0_g.sh
sbatch slurm/train_mpatch_v0_h.sh
