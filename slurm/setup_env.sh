#!/bin/bash
# Run once interactively on the BYU HPC login node — do NOT submit via sbatch.
# Install the fibrin conda environment from environment.yml.
module load miniforge3
mamba env create -f environment.yml
