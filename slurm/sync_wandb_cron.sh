#!/bin/bash
# Scheduled wandb sync — managed by crontab on the BYU HPC login node.
# Do NOT run via sbatch; install with: bash slurm/install_wandb_cron.sh
#
# Reads expiry date from slurm/wandb_cron_expiry.txt and auto-removes
# this cron entry once that date passes.

REPO_DIR="$(cd "$(dirname "$(realpath "$0")")/.." && pwd)"
WANDB="$HOME/.conda/envs/fibrin/bin/wandb"
LOG="$REPO_DIR/slurm/logs/wandb_sync_cron.log"
EXPIRY_FILE="$REPO_DIR/slurm/wandb_cron_expiry.txt"
EXPIRY_DATE="$(cat "$EXPIRY_FILE" 2>/dev/null || echo "2099-01-01")"

mkdir -p "$REPO_DIR/slurm/logs"

if [[ "$(date +%Y-%m-%d)" > "$EXPIRY_DATE" ]]; then
    echo "$(date): Expired ($EXPIRY_DATE) — removing cron entry." >> "$LOG"
    crontab -l 2>/dev/null | grep -v "sync_wandb_cron.sh" | crontab -
    exit 0
fi

echo "$(date): Syncing wandb runs..." >> "$LOG"
COUNT=0
while IFS= read -r -d '' dir; do
    # No --no-mark-synced: let wandb mark each directory after upload so it is
    # not re-uploaded on the next cron tick.  Re-uploading old job directories
    # can cause the wandb server to reset the run history and discard continuation
    # data from later jobs.  Use sync_wandb.sh (--no-mark-synced) to force a
    # manual re-sync when needed.
    if "$WANDB" sync "$dir" >> "$LOG" 2>&1; then
        echo "  Synced: $dir" >> "$LOG"
        COUNT=$((COUNT + 1))
    fi
done < <(find "$REPO_DIR" -maxdepth 4 -type d \
         \( -name "offline-run-*" -o -name "run-*" \) \
         -path "*/wandb/*" -print0)
echo "$(date): Done — $COUNT run(s) synced." >> "$LOG"
