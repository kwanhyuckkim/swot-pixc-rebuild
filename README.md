# swot-pixc-rebuild

Rebuilding the SWOT pixel cloud (PIXC) from the L1B HR SLC data, plus a denser version of it. The input is one L1B_HR_SLC tile and the L2_HR_PIXC of the same tile.

Each PIXC pixel is the mean of 7 SLC lines at one range sample. The code recomputes that mean from the SLC and classifies it with its own detector (the "reproduced" PIXC). It also computes the same 7-line mean centered on every SLC line ("densified"). A densified cell is filled only where at least 4 of the 7 lines in its window have values, so it never fills lines outside the pixels PIXC saved. Each value still averages about 7 lines (21.9 m on tile 037L), so you get more samples but the same resolution.

The other outputs are the single-look power and a block-filled version that copies each reproduced pixel onto its 7 lines. There is also an optional comparison with RiverScope water masks (Planet, 3 m). The code does not separate bright bars from water; that is left to your own classifier.

## Files

- `densepixc/dpc_core.py`: shared functions
- `densepixc/dpc_run_tile.py`: runs one tile
- `densepixc/dpc_plot.py`: figures
- `densepixc/make_rs_json.py`: builds the RiverScope file list
- `densepixc/dpc_*.sbatch`: Slurm jobs for tiles, figures and tests
- `densepixc/tests/`: unit tests

## How to run

Without Slurm:

```bash
cd densepixc
python dpc_run_tile.py --slc SLC.nc --pixc PIXC.nc --out out/dpc_TILE.nc [--riverscope rs.json] [--threads 4]
python dpc_plot.py --dense out/dpc_TILE.nc --riverscope rs.json --windows windows.json --outdir out/figures
```

On a Slurm cluster, make a `logs/` folder inside `densepixc/` first, then:

```bash
sbatch dpc_tiles.sbatch tiles.txt                  # first tile in the list
sbatch --array=1-N dpc_tiles.sbatch tiles.txt      # N tiles, one task each
sbatch dpc_test.sbatch                             # unit tests
```

`tiles.txt` has one tile per line: `TILE SLC.nc PIXC.nc OUTDIR [RiverScope list JSON or -] [windows JSON or -]`. The sbatch files default to the Python in my conda environment on Unity, so pass `--export=ALL,PY=/path/to/python` to use yours.

Packages: numpy, scipy, matplotlib, netCDF4, pyproj, and rasterio for the RiverScope part.
