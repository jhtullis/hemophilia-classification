#!/bin/bash
# Run once interactively on the BYU HPC login node — do NOT submit via sbatch.
# Install the fibrin conda environment from environment.yml.
module load miniforge3
mamba env create -f environment.yml

# After env creation, authenticate Weights & Biases (one-time per user).
# Add your API key to ~/.bashrc so all jobs pick it up automatically:
#
#   echo 'export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>' >> ~/.bashrc
#   source ~/.bashrc
#
# BYU HPC compute nodes have no outbound internet, so slurm jobs run wandb
# in offline mode (WANDB_MODE=offline is set in each job script). After a job
# completes, sync from the login node (which has internet):
#
#   bash slurm/sync_wandb.sh
