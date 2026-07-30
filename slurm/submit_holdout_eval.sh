#!/bin/bash
# Submit both holdout-evaluation pipelines. The lite_lc aggregation job runs
# only after the lite_lc batch eval completes successfully (afterok).
#
# Usage (from project root):
#   bash slurm/submit_holdout_eval.sh
set -e
mkdir -p slurm/logs

sbatch slurm/eval_mpatch_v1e_125s_ensemble.sh

LC_JOB_ID=$(sbatch --parsable slurm/eval_mpatch_v1e_lite_lc.sh)
sbatch --dependency=afterok:${LC_JOB_ID} slurm/eval_mpatch_v1e_lite_lc_aggregate.sh
echo "Submitted lite_lc eval (job ${LC_JOB_ID}) with aggregate job dependent on success."
