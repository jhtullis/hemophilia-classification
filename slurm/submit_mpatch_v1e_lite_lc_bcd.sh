#!/bin/bash
# Submit all mpatch_v1e_lite b/c/d learning curve repetition jobs.
# Prerequisite: mpatch_v0e_lite must have completed at least one epoch so that
# models/mpatch_v0e_lite/train_record_patch.json exists.

PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

SPLIT_RECORD="${PROJECT_DIR}/models/mpatch_v0e_lite/train_record_patch.json"
if [ ! -f "${SPLIT_RECORD}" ]; then
    echo "ERROR: split record not found: ${SPLIT_RECORD}"
    echo "Train mpatch_v0e_lite at least one epoch before submitting LC jobs."
    exit 1
fi

for series in b c d; do
    for suffix in lc05 lc10 lc15 lc20 lc25 lc30 lc35; do
        sbatch "${PROJECT_DIR}/slurm/train_mpatch_v1e_lite_${suffix}${series}.sh"
        echo "Submitted mpatch_v1e_lite_${suffix}${series}"
    done
done
