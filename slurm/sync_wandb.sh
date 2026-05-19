#!/bin/bash
# Run interactively on the BYU HPC login node — do NOT submit via sbatch.
# Syncs all offline wandb runs (all model types) to the dashboard.
# Safe to re-run any time; --no-mark-synced keeps local data intact.
module load miniforge3
conda activate fibrin
for dir in models/*/wandb/run-*/; do
    [ -d "$dir" ] && wandb sync "$dir" --no-mark-synced && echo "Synced: $dir"
done
echo "Done. Check your dashboard at wandb.ai"
