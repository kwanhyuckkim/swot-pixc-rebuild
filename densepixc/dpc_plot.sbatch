#!/bin/bash -l
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=24G
#SBATCH -t 00:45:00
#SBATCH -J dpc_plot
#SBATCH -o logs/%x_%j.out
# Re-draw window figures from an existing dense file (no SLC needed):
#   sbatch dpc_plot.sbatch out/dpc_037L.nc out/rs_037L.json examples/windows_037L.json out/figures_037L
#   or --export=ALL,DENSE=...,RSL=...,WINDOWS=...,OUTDIR=...
set -euo pipefail
export PYTHONNOUSERSITE=1
PY=${PY:-/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python}
cd "${SLURM_SUBMIT_DIR:-.}"
"$PY" -u dpc_plot.py --dense "${1:-${DENSE:?dense nc}}" --riverscope "${2:-${RSL:?riverscope list json}}" --windows "${3:-${WINDOWS:?windows json}}" --outdir "${4:-${OUTDIR:?output folder}}"
