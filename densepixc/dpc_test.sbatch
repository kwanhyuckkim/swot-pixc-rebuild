#!/bin/bash -l
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -t 00:10:00
#SBATCH -J dpc_test
#SBATCH -o logs/%x_%j.out
# Unit tests only (synthetic data, about 10 s):  sbatch dpc_test.sbatch
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=${PY:-/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python}
cd "${SLURM_SUBMIT_DIR:-.}"
"$PY" -u tests/test_dpc_core.py
