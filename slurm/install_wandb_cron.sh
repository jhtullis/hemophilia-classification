#!/bin/bash
# Run once interactively on the BYU HPC login node to install the wandb sync cron job.
# To remove early: crontab -l | grep -v sync_wandb_cron | crontab -

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CRON_SCRIPT="$REPO_DIR/slurm/sync_wandb_cron.sh"
EXPIRY_FILE="$REPO_DIR/slurm/wandb_cron_expiry.txt"
EXPIRY="$(date -d '+7 days' +%Y-%m-%d)"

# Write expiry date
echo "$EXPIRY" > "$EXPIRY_FILE"

# Add cron entry (every 3 hours) — skip if already installed
if crontab -l 2>/dev/null | grep -q "sync_wandb_cron.sh"; then
    echo "Cron entry already exists — updating expiry to $EXPIRY."
else
    (crontab -l 2>/dev/null; echo "*/15 * * * * bash $CRON_SCRIPT") | crontab -
    echo "Cron job installed: runs every 15 minutes until $EXPIRY."
fi

echo "Log file: $REPO_DIR/slurm/logs/wandb_sync_cron.log"
echo "To remove early: crontab -l | grep -v sync_wandb_cron | crontab -"
