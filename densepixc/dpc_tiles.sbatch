#!/bin/bash -l
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -J dpc_tile
#SBATCH -o logs/%x_%j.out
# densepixc: one SWOT tile per task (CPU). Submit from the densepixc folder (logs/ must exist):
#   sbatch dpc_tiles.sbatch tiles.txt                                  # first tile of the list
#   sbatch --array=1-N dpc_tiles.sbatch tiles.txt                      # N tiles, one task each (task numbers start at 1)
# tiles.txt, one tile per line:  TILE  SLC.nc  PIXC.nc  OUTDIR  [RiverScope list JSON or -]  [windows JSON or -]
#   TILE is a unique label used in the output names; text after '#' is ignored.
# The list can also come as --export=ALL,TILES=tiles.txt (needed when a wrapper puts sbatch options after the script).
# Optional (via --export=ALL,...):
#   RS_FILES=aux/rs_files.json  build the RiverScope list with make_rs_json.py when the line's RiverScope column is '-' (TILE must be a key there)
#   PY=/path/to/python          default: the swot-urban conda env of the author (needs numpy scipy matplotlib netCDF4 pyproj rasterio)
#   NOIFG=1                     do not write per-line powers and interferogram (smaller output)
#   GATES=warn                  quality gates D, F and the placed share warn instead of stopping the run
#   CLASSES=pixc                water/land classes copied from the PIXC (JPL) instead of our detector (default CLASSES=ours)
#   DENSIFY=landstop            densified moving average that stops at land lines (default DENSIFY=plain)
set -euo pipefail
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
PY=${PY:-/work/kwanhyuckkim_umass_edu/.conda/envs/swot-urban/bin/python}
[ -x "$PY" ] || { echo "python not found: $PY (set PY=/path/to/python with --export)"; exit 2; }
cd "${SLURM_SUBMIT_DIR:-.}"
LIST=${1:-${TILES:-}}
[ -n "$LIST" ] && [ -f "$LIST" ] || { echo "tile list not found: '$LIST' (sbatch dpc_tiles.sbatch tiles.txt, or --export=ALL,TILES=tiles.txt)"; exit 2; }
LINES=$(sed -e 's/\r$//' -e 's/#.*//' "$LIST" | awk 'NF')                     # drop comments, blank lines and Windows line ends
N=$(printf '%s\n' "$LINES" | awk 'NF' | wc -l | tr -d ' ')
i=${SLURM_ARRAY_TASK_ID:-1}
[ "$N" -ge 1 ] || { echo "no tile in $LIST"; exit 2; }
[ "$i" -ge 1 ] && [ "$i" -le "$N" ] || { echo "task $i but $LIST has tiles 1..$N (use --array=1-$N)"; exit 2; }
DUP=$(printf '%s\n' "$LINES" | awk '{print $1}' | sort | uniq -d)
[ -z "$DUP" ] || { echo "TILE labels must be unique (they name the outputs); repeated: $DUP"; exit 2; }
LINE=$(printf '%s\n' "$LINES" | sed -n "${i}p")
read -r TILE SLC PIXC OUTDIR RS WIN _ <<< "$LINE"
RS=${RS:--}; WIN=${WIN:--}
[ -n "${OUTDIR:-}" ] || { echo "line $i needs at least: TILE SLC PIXC OUTDIR"; exit 2; }
for f_ in "$SLC" "$PIXC"; do [ -f "$f_" ] || { echo "file not found: $f_"; exit 2; }; done
[ "$RS" = "-" ] || [ -f "$RS" ] || { echo "RiverScope list not found: $RS"; exit 2; }
[ "$WIN" = "-" ] || [ -f "$WIN" ] || { echo "windows file not found: $WIN"; exit 2; }
[ -z "${RS_FILES:-}" ] || [ -f "$RS_FILES" ] || { echo "RS_FILES not found: $RS_FILES"; exit 2; }
mkdir -p "$OUTDIR"
echo "tile $TILE, task $i of $N, $(hostname), $(date)"
"$PY" -u tests/test_dpc_core.py                                 # unit tests first: a broken install stops here
if [ -n "${RS_FILES:-}" ] && [ "$RS" = "-" ]; then
  rc=0; "$PY" -u make_rs_json.py --rs-files "$RS_FILES" --tile "$TILE" --out "$OUTDIR/rs_$TILE.json" || rc=$?
  case $rc in
    0) RS="$OUTDIR/rs_$TILE.json" ;;
    3) echo "no RiverScope record for $TILE: RiverScope matching and figures are skipped" ;;
    *) echo "make_rs_json.py failed (exit $rc)"; exit "$rc" ;;
  esac
fi
OPT=(--slc "$SLC" --pixc "$PIXC" --out "$OUTDIR/dpc_$TILE.nc" --gates "${GATES:-stop}" --classes "${CLASSES:-ours}" --densify "${DENSIFY:-plain}")
[ "$RS" = "-" ] || OPT+=(--riverscope "$RS")
[ -z "${NOIFG:-}" ] || OPT+=(--no-ifg)
"$PY" -u dpc_run_tile.py "${OPT[@]}"
if [ "$WIN" != "-" ]; then
  if [ "$RS" != "-" ]; then "$PY" -u dpc_plot.py --dense "$OUTDIR/dpc_$TILE.nc" --riverscope "$RS" --windows "$WIN" --outdir "$OUTDIR/figures_$TILE"
  else echo "windows given but no RiverScope list: figures skipped"; fi
fi
echo "done $(date)"
