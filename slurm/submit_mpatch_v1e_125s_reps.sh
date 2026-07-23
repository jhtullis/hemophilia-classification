#!/bin/bash
# Submit all seven mpatch_v1e_125s cross-validation replicates (rep_a..rep_g).
# mpatch_v1e_125s itself is submitted separately via slurm/submit_mpatch_v1e_125s.sh.
# Requires mpatch_v1e_125s to have already produced models/mpatch_v1e_125s/
# train_record_patch.json (true as of this writing -- it is currently training).
#
# Usage (from project root):
#   bash slurm/submit_mpatch_v1e_125s_reps.sh
mkdir -p slurm/logs
sbatch slurm/train_mpatch_v1e_125s_rep_a.sh
sbatch slurm/train_mpatch_v1e_125s_rep_b.sh
sbatch slurm/train_mpatch_v1e_125s_rep_c.sh
sbatch slurm/train_mpatch_v1e_125s_rep_d.sh
sbatch slurm/train_mpatch_v1e_125s_rep_e.sh
sbatch slurm/train_mpatch_v1e_125s_rep_f.sh
sbatch slurm/train_mpatch_v1e_125s_rep_g.sh
