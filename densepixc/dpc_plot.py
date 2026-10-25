#!/usr/bin/env python3
"""densepixc - window figures from a dense pixel cloud file: SLC | existing PIXC | reproduced PIXC | block-filled | densified | RiverScope,
and below each product its disagreement with RiverScope on a 1 m grid.

Scores (also in scores.json): missed = RiverScope water not drawn as water, as a share of RiverScope water (it includes water with no pixel drawn at
all, shown in its own colour and given as missed_no_pixel); false = RiverScope land drawn as water, as a share of the drawn water. Only cloud-free,
labelled RiverScope pixels count.
Drawing: our cells (one SLC line x one range sample) are quadrilaterals around their centres at the water level; neighbouring cells share corners
(dpc_core.cell_mesh), so nothing is left blank between them. PIXC pixels (existing and reproduced) are rectangles centred on the PIXC position,
one PIXC azimuth spacing long along track and pixel_area / spacing wide. Where footprints overlap, water is painted last.
A window that fails (e.g. outside the tile) is reported in scores.json and the run ends with an error after the other windows are drawn.
Usage:  python dpc_plot.py --dense dpc_TILE.nc --riverscope rs.json --windows windows.json --outdir figs/
        windows.json = [{"name": "W01", "x0": 564336, "y0": 7262095, "size": 500}, ...] in the tile's UTM zone (attribute utm_epsg of the dense file)"""
import argparse, json, os, sys, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpc_core as C

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--dense", required=True); ap.add_argument("--riverscope", required=True); ap.add_argument("--windows", required=True); ap.add_argument("--outdir", required=True)
ap.add_argument("--margin", type=float, default=80.0, help="metres around the window from which cells/pixels are drawn")
args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)
import netCDF4 as nc, pyproj
d = nc.Dataset(args.dense); d.set_auto_mask(False); gc, gp = d["cells"], d["pixc"]; EPSG = int(d.utm_epsg); L = int(d.num_azimuth_looks)
CSRC = {"ours": "our detector", "pixc": "PIXC (JPL)"}.get(str(getattr(d, "class_source", "ours")), "our detector")   # who decided the classes of products 3-5
DSRC = {"plain": "plain 7-line moving average", "landstop": "7-line moving average that stops at land lines"}.get(str(getattr(d, "densify", "plain")), "plain 7-line moving average")
to_utm = pyproj.Transformer.from_crs(4326, EPSG, always_xy=True); to_geo = pyproj.Transformer.from_crs(EPSG, 4326, always_xy=True)
def utm(lon, lat):
    x, y = to_utm.transform(lon, lat); x = np.asarray(x, float); y = np.asarray(y, float); bad = ~(np.isfinite(x) & np.isfinite(y)); x[bad] = np.nan; y[bad] = np.nan; return x, y
CX, CY = utm(gc["longitude"][:], gc["latitude"][:]); PX, PY = utm(gp["longitude"][:], gp["latitude"][:])
CK = gc["slc_line"][:].astype(np.int64); CR = gc["range_index"][:].astype(np.int64)
CDB = gc["power_db"][:]; CREP = gc["class_reproduced"][:]; CDEN = gc["class_densified"][:]
PEX = gp["class_existing"][:]; PREP = gp["class_reproduced"][:]; PAREA = gp["pixel_area"][:].astype(float)
with open(args.riverscope) as f: recs = json.load(f)
with open(args.windows) as f: WIN = json.load(f)
names = [str(w.get("name")) for w in WIN]; C.check(len(set(names)) == len(names), f"window names must be unique: {names}")
# PIXC classes 1-7 (0 = no pixel), the colours of the study's figures; names from the PIXC flag attributes copied into the dense file
CCOL = np.array([[255, 255, 255], [217, 217, 217], [216, 182, 122], [158, 202, 225], [33, 113, 181], [106, 61, 154], [65, 182, 196], [8, 48, 107]]) / 255.0
_cv = gc["class_existing"]; _at = _cv.ncattrs()
CNAME = {1: "land", 2: "land_near_water", 3: "water_near_land", 4: "open_water", 5: "dark_water", 6: "low_coh_water_near_land", 7: "open_low_coh_water"}
if "flag_values" in _at and "flag_meanings" in _at and len(np.atleast_1d(_cv.flag_values)) == len(str(_cv.flag_meanings).split()):
    CNAME = {int(v_): m_ for v_, m_ in zip(np.atleast_1d(_cv.flag_values), str(_cv.flag_meanings).split()) if 1 <= int(v_) <= 7}
print("class names:", CNAME, flush=True)
WCOL = CCOL[4]; LCOL = CCOL[1]                                                           # RiverScope water / land
from matplotlib.colors import LinearSegmentedColormap
SLC_CMAP = LinearSegmentedColormap.from_list("slc", [(0, 0, 0), (0.8, 0.8, 0.8)])     # black to light grey, so white stays 'no cell'
# bottom row: 0 no RiverScope label, 1 RiverScope land and no water drawn, 2 both water, 3 missed (drawn as land), 4 missed (no pixel), 5 false water
DC = np.array([[1, 1, 1], [0.86, 0.86, 0.86], [0.55, 0.75, 0.90], [0.85, 0.15, 0.10], [0.96, 0.66, 0.70], [0.98, 0.60, 0.10]])
def pc_(v): return "–" if not np.isfinite(v) else f"{v:.0f}%"
SCORES = {}; FAILED = []
for w in WIN:
    name = str(w.get("name"))
    try:
        x0, y0, S = float(w["x0"]), float(w["y0"]), float(w.get("size", 500)); n = int(round(S)); M = args.margin
        sel = np.where((CX > x0 - M) & (CX < x0 + S + M) & (CY > y0 - M) & (CY < y0 + S + M))[0]
        C.check(len(sel) >= 1000, f"{name}: only {len(sel)} placed cells in the window (outside this tile?)")
        f, res = C.fit_cell_geometry(CK[sel], CR[sel], CX[sel] - x0, CY[sel] - y0)
        quads = C.cell_mesh(CX[sel] - x0, CY[sel] - y0, CK[sel], CR[sel], f)
        km, rm = float(np.median(CK[sel])), float(np.median(CR[sel])); ax_, ay_ = f(km + 0.5, rm); bx_, by_ = f(km - 0.5, rm); along = np.array([ax_ - bx_, ay_ - by_])
        cx_, cy_ = f(km, rm + 0.5); dx_, dy_ = f(km, rm - 0.5); line_m, sample_m = float(np.hypot(*along)), float(np.hypot(cx_ - dx_, cy_ - dy_))
        C.check(1.0 < line_m < 10.0 and 1.0 < sample_m < 200.0, f"{name}: implausible cell size {line_m:.2f} m along track x {sample_m:.2f} m across")
        spacing = float(L * line_m); psel = np.where((PX > x0 - M) & (PX < x0 + S + M) & (PY > y0 - M) & (PY < y0 + S + M))[0]
        pseudo = C.pseudo_pixels(PX[psel] - x0, PY[psel] - y0, PAREA[psel], along, spacing)
        g = np.arange(n) + 0.5; GX, GY = np.meshgrid(x0 + g, y0 + g); lo_, la_ = to_geo.transform(GX.ravel(), GY.ravel())
        rs = C.riverscope_sample(np.asarray(la_), np.asarray(lo_), recs).reshape(n, n)                   # RiverScope on the 1 m grid
        def paint(polys, cls):                                                          # class raster (NaN = no pixel), water painted last
            cls = np.asarray(cls).astype(float); o = np.argsort(np.isin(cls, C.WATER), kind="stable"); return C.rasterize(polys, cls, n, order=o)
        def by_class(v, mask):
            vi = np.nan_to_num(v, nan=0.0).astype(int); return {int(c): int(((vi == c) & mask).sum()) for c in range(1, 8) if ((vi == c) & mask).any()}
        def keep(polys, cls):                                                           # class 0 = no value: not drawn
            ok_ = np.asarray(cls) > 0; return polys[ok_], np.asarray(cls)[ok_]
        LAY = {"existing": paint(*keep(pseudo, PEX[psel])), "reproduced": paint(*keep(pseudo, PREP[psel])), "block": paint(*keep(quads, CREP[sel])), "densified": paint(*keep(quads, CDEN[sel]))}
        slc = C.rasterize(quads, CDB[sel], n, order=np.arange(len(sel))); lo, hi = np.nanpercentile(slc, [2, 98])
        WAT = {k: np.isin(np.nan_to_num(v, nan=0.0).astype(int), C.WATER) for k, v in LAY.items()}; okr = rs >= 0
        SC = {k: C.disagreement(WAT[k], rs, drawn=np.isfinite(v)) for k, v in LAY.items()}
        for k, v in LAY.items():                                                       # which classes make the false and the missed water
            SC[k]["false_m2_by_class"] = by_class(v, okr & (rs == 0) & WAT[k]); SC[k]["missed_drawn_as_land_m2_by_class"] = by_class(v, okr & (rs == 1) & ~WAT[k] & np.isfinite(v))
        SCORES[name] = dict(SC, geometry=dict(fit_residual_max_m=res, line_m=line_m, sample_m=sample_m, pseudo_pixel_length_m=spacing, cells=int(len(sel)), pixc_pixels=int(len(psel))))
        fig, axs = plt.subplots(2, 6, figsize=(24, 8.9), dpi=110); ext = [0, S, 0, S]
        axs[0, 0].imshow(slc, origin="lower", extent=ext, cmap=SLC_CMAP, vmin=lo, vmax=hi, interpolation="nearest"); axs[0, 0].set_title("SLC")
        TT = {"existing": "Existing PIXC", "reproduced": "Reproduced PIXC", "block": "Block-filled", "densified": "Densified"}
        for j, key in enumerate(("existing", "reproduced", "block", "densified"), start=1):
            v = LAY[key]; vi = np.clip(np.nan_to_num(v, nan=0.0).astype(int), 0, 7); axs[0, j].imshow(CCOL[vi], origin="lower", extent=ext, interpolation="nearest"); axs[0, j].set_title(TT[key])
            wat = WAT[key]; drawn = np.isfinite(v); ok = rs >= 0; dimg = np.zeros((n, n), int)
            dimg[ok & (rs == 0) & ~wat] = 1; dimg[ok & (rs == 1) & wat] = 2; dimg[ok & (rs == 1) & ~wat & drawn] = 3; dimg[ok & (rs == 1) & ~drawn] = 4; dimg[ok & (rs == 0) & wat] = 5
            axs[1, j].imshow(DC[dimg], origin="lower", extent=ext, interpolation="nearest"); axs[1, j].set_title(f"missed {pc_(SC[key]['missed_pct'])}, false {pc_(SC[key]['false_pct'])}")
        img = np.ones((n, n, 3)); img[rs == 0] = LCOL; img[rs == 1] = WCOL; axs[0, 5].imshow(img, origin="lower", extent=ext, interpolation="nearest"); axs[0, 5].set_title("RiverScope")
        for a in axs.ravel(): a.set_xlim(0, S); a.set_ylim(0, S); a.set_aspect("equal"); a.set_xticks([0, S / 2, S]); a.set_yticks([0, S / 2, S])
        for a in (axs[1, 0], axs[1, 5]): a.axis("off")
        edge = dict(edgecolor="0.6", linewidth=0.6)
        axs[1, 0].legend(handles=[Patch(facecolor=CCOL[c], label=f"{c} {CNAME.get(c, '').replace('_', ' ')}", **(edge if c == 1 else dict(edgecolor=CCOL[c]))) for c in range(1, 8)]
                         + [Patch(facecolor="white", label="no pixel", **edge)], loc="center", frameon=False, fontsize=10.5, title="PIXC class", title_fontsize=11)
        axs[1, 5].legend(handles=[Patch(color=DC[2], label="both water"), Patch(color=DC[1], label="RiverScope land,\nno water drawn"), Patch(color=DC[3], label="missed: drawn as land"),
                                  Patch(color=DC[4], label="missed: no pixel"), Patch(color=DC[5], label="false water"), Patch(facecolor="white", label="no RiverScope label", **edge)],
                         loc="center", frameon=False, fontsize=10.5, title="vs RiverScope", title_fontsize=11)
        fig.text(0.01, 0.035, f"{name}, metres from ({x0:.0f} E, {y0:.0f} N), EPSG:{EPSG}. Classes of reproduced, block-filled and densified: {CSRC}. Densified: {DSRC}.", fontsize=10)
        fig.text(0.01, 0.012, "PIXC pixels at their PIXC positions; SLC, block-filled and densified cells at the water level. "
                 "Missed: % of RiverScope water; false: % of drawn water.", fontsize=10)
        fig.tight_layout(rect=[0, 0.06, 1, 1]); fig.savefig(os.path.join(args.outdir, f"{name}.png")); plt.close(fig)
        print(name, {k: (v["wrong_m2"], round(v["missed_pct"], 1), round(v["false_pct"], 1), round(v["missed_no_pixel_pct"], 1)) for k, v in SC.items()}, flush=True)
    except Exception as e:                                                    # record and go on; the run still fails at the end
        plt.close("all"); FAILED.append(name); SCORES[name] = dict(failed=f"{type(e).__name__}: {e}"); print(f"FAILED {name}: {type(e).__name__}: {e}", flush=True); traceback.print_exc()
with open(os.path.join(args.outdir, "scores.json"), "w") as f: json.dump(SCORES, f, indent=1, default=float)
C.check(not FAILED, f"{len(FAILED)} of {len(WIN)} windows failed: {FAILED} (details in scores.json and above)")
