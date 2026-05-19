#!/bin/bash
# Run once interactively on the BYU HPC login node — do NOT submit via sbatch.
# Install the fibrin conda environment from environment.yml.
module load miniforge3
mamba env create -f environment.yml

# After env creation, authenticate Weights & Biases (one-time per user):
#
#   conda activate fibrin && wandb login
#
# Alternatively, set your API key in ~/.bashrc or in each job script:
#   export WANDB_API_KEY=<your-key-from-wandb.ai/authorize>
#
# To run jobs without internet access (sync later):
#   export WANDB_MODE=offline
#   # After the job completes, sync the run:
#   wandb sync models/<type>/wandb/run-*/
