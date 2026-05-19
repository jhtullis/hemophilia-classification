#!/bin/bash
# Submit all four training jobs. They are independent and can run concurrently
# if multiple GPUs are available.
sbatch slurm/train_5class.sh
sbatch slurm/train_5class_hpc_v0.sh
sbatch slurm/train_3class_scratch.sh
sbatch slurm/train_3class_finetune.sh
