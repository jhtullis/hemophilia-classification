#!/bin/bash
# Submit all mpatch_v1e_lite learning curve jobs.
# Prerequisite: mpatch_v0e_lite must have completed at least one epoch so that
# models/mpatch_v0e_lite/train_record_patch.json exists.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

SPLIT_RECORD="${PROJECT_DIR}/models/mpatch_v0e_lite/train_record_patch.json"
if [ ! -f "${SPLIT_RECORD}" ]; then
    echo "ERROR: split record not found: ${SPLIT_RECORD}"
    echo "Train mpatch_v0e_lite at least one epoch before submitting LC jobs."
    exit 1
fi

for suffix in lc05a lc10a lc15a lc20a lc25a lc30a lc35a; do
    sbatch "${PROJECT_DIR}/slurm/train_mpatch_v1e_lite_${suffix}.sh"
    echo "Submitted mpatch_v1e_lite_${suffix}"
done
