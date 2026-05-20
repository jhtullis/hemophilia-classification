#!/bin/bash
# Run interactively on the BYU HPC login node — do NOT submit via sbatch.
# Syncs all offline wandb runs (all model types) to the dashboard.
# Safe to re-run any time; --no-mark-synced keeps local data intact.
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WANDB="$HOME/.conda/envs/fibrin/bin/wandb"
while IFS= read -r -d '' dir; do
    "$WANDB" sync "$dir" --no-mark-synced && echo "Synced: $dir"
done < <(find "$REPO_DIR" -maxdepth 4 -type d \
         \( -name "offline-run-*" -o -name "run-*" \) \
         -path "*/wandb/*" -print0)
echo "Done. Check your dashboard at wandb.ai"
