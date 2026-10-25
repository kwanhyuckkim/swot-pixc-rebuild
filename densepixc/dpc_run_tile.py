#!/usr/bin/env python3
"""densepixc - one SWOT tile: L1B_HR_SLC + L2_HR_PIXC (+ optional RiverScope labels) -> dense pixel cloud (netCDF) + checks (JSON).

Products, as water/land classes on every SLC line of every saved PIXC pixel ('cell' = one SLC line x one range sample):
  existing   - the PIXC classification (JPL), copied to the 7 lines of its PIXC pixel
  reproduced - 7-line multilook as JPL does it (mean over the 7 SLC lines of a PIXC pixel = one value per pixel), classified by our detector
  block      - the reproduced value/class written to all 7 lines of its pixel (same information as 'reproduced', no gaps)
  densified  - sliding 7-line multilook centred on every SLC line (moving average along azimuth, no decimation), classified on the SLC-line grid;
               a cell is filled only when at least 4 of the 7 window lines have values (always true inside a pixel with all 7 lines)
Every cell is placed on the ground at the water level (median PIXC water height, 100 m grid) and matched to RiverScope at that position.
Cells whose range sphere cannot reach the water level (terrain near nadir) keep NaN latitude/longitude.
Water = classes 3-7. Detection = gamma likelihood + Ising MRF (see dpc_core.detect); class 5 (dark water) is kept from the PIXC.

Built-in checks stop the run (dpc_core.CheckError) and still write <out>_checks.json with status "failed: ...":
  A rare powers from the SLC = PIXC power_plus_y / power_minus_y (median ratio within 1e-4 of 1, 1st-99th percentile within 1e-3): pairing, line offset
  B geolocation at the PIXC height = PIXC latitude/longitude (median < 2 m, 99th percentile < 5 m; points the PIXC put beyond nadir are left out)
  C PIXC interferogram phase = our 7-line interferogram + GrDEM reference phase (median |residual| < 0.05 rad after one constant)
  D coherent power: after one constant, the predicted phase explains the interior-water phase (resultant > 0.2), and the coherent term raises
    water more than land. It catches prediction errors that vary across the scene, not an error that is nearly constant over the water.
  E sliding multilook = reproduced multilook on the centre lines of pixels with all 7 lines (1st and 99th percentile of the ratio within 1e-5 of 1)
  F our detection agrees with the PIXC water/land on >= 80 % of the decided elements (sanity, not a target)
D and F, and the share of cells placed (> 50 %), are quality gates: an unusual scene (very low water coherence, mostly terrain near nadir) can stop
a correct run; the checks JSON then says which gate failed. With --gates warn the product is written anyway, with status passed_with_warnings.
<out>_checks.json says 'running' while the job runs, 'failed: ...' after an error, a time limit or scancel, and 'passed' at the end.
Water/land classes of products 3-5: --classes ours (default) runs our detector on the new values; --classes pixc copies the PIXC (JPL) classes
(then block-filled and densified have the same classes, and check F is skipped).
Moving average of the densified product: --densify plain (default) or landstop (the window slides away from land lines; user rule 10-08).
Usage:  python dpc_run_tile.py --slc SLC.nc --pixc PIXC.nc --out dpc_TILE.nc [--riverscope rs.json] [--no-ifg] [--threads N] [--gates warn] [--classes pixc] [--densify landstop]
        rs.json = [{"label": label.tif, "sr": planet_sr.tif or null, "udm2": planet_udm2.tif or null}, ...] (the first record with a usable label wins)"""
import argparse, json, os, resource, signal, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dpc_core as C

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--slc", required=True); ap.add_argument("--pixc", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--riverscope", default=None, help="JSON list of RiverScope label records (optional)")
ap.add_argument("--no-ifg", action="store_true", help="do not write per-line powers and interferogram (smaller file)")
ap.add_argument("--chunk", type=int, default=1_000_000, help="cells per geolocation chunk")
ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")), help="threads for the placement step (default: SLURM_CPUS_PER_TASK or 1)")
ap.add_argument("--gates", choices=("stop", "warn"), default="stop", help="quality gates D, F and the placed share: stop the run (default) or warn and go on")
ap.add_argument("--densify", choices=("plain", "landstop"), default="plain", help="densified moving average: plain 7-line window (default), or "
                "landstop: the window slides away from land (class 1) lines and land lines keep their own value (user rule 10-08)")
ap.add_argument("--classes", choices=("ours", "pixc"), default="ours", help="water/land classes of the reproduced, block-filled and densified products: "
                "ours = our detector on the new values (default); pixc = the PIXC (JPL) classes copied to every line")
args = ap.parse_args()
t0 = time.time(); TIM = {}; CHK = {"slc_file": os.path.basename(args.slc), "pixc_file": os.path.basename(args.pixc)}
JPATH = os.path.splitext(args.out)[0] + "_checks.json"
def log(*a): print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
def stage(name, t_start): TIM[name] = round(time.time() - t_start, 1); log(f"{name} done in {TIM[name]} s")
def pct(v, q):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; return [float("nan")] * len(q) if v.size == 0 else [float(x) for x in np.percentile(v, q)]
def dump(status):                                       # checks, timings and memory; also written when a check fails
    CHK["status"] = status; CHK["timing_s"] = dict(TIM, total=round(time.time() - t0, 1))
    CHK["peak_memory_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3 if sys.platform == "darwin" else 1024 ** 2), 2)
    os.makedirs(os.path.dirname(os.path.abspath(JPATH)), exist_ok=True)
    with open(JPATH, "w") as f: json.dump(CHK, f, indent=1, default=float)
_hook0 = sys.excepthook
def _hook(tp, val, tb):
    try: dump(f"failed: {tp.__name__}: {val}")
    except Exception: pass
    _hook0(tp, val, tb)
sys.excepthook = _hook
def _term(signum, frame):                                # Slurm time limit, scancel, preemption: say so in the checks file
    dump(f"failed: signal {signum} (time limit, scancel or preemption)"); sys.exit(128 + signum)
signal.signal(signal.SIGTERM, _term)
WARN = []
def gate(cond, msg):                                     # quality gate: CheckError, or a warning with --gates warn
    if cond: return
    C.check(args.gates == "warn", msg); WARN.append(msg); log("WARNING", msg)
dump("running")

# ---------------------------------------------------------------- inputs and cells
RS_RECS = None
if args.riverscope:                                                      # check the RiverScope list now, not after the slow steps
    C.check(os.path.exists(args.riverscope), f"RiverScope list not found: {args.riverscope}")
    with open(args.riverscope) as f: RS_RECS = json.load(f)
    C.check(isinstance(RS_RECS, list) and len(RS_RECS) > 0, f"{args.riverscope}: expected a non-empty JSON list")
    for r_ in RS_RECS:
        C.check(bool(r_.get("label")), f"{args.riverscope}: a record has no 'label'")
        for key in ("label", "sr", "udm2"):
            if r_.get(key): C.check(os.path.exists(r_[key]), f"RiverScope file not found: {r_[key]}")
    import rasterio  # noqa: F401  (used by dpc_core.riverscope_sample; fail early when it is missing)
s0 = time.time(); t = C.load_tile(args.slc, args.pixc); npix = len(t.ai); CHK["pixc_fill_values"] = t.fill_counts
log(f"tile {t.tile_name}: SLC {t.NL} lines x {t.NP} samples; PIXC {npix:,} pixels, {t.NA} rare lines; L {t.L}, OFF {t.OFF}, SF {t.SF}, EPSG {t.epsg}")
log("PIXC fill values (set to NaN / class 0)", {k: v for k, v in t.fill_counts.items() if v})
K, RR, PID, D = C.dense_cells(t.ai, t.ri, t.L, t.OFF, t.NL); ncell = len(K); A = t.ai[PID]; CHK["cells"] = ncell; CHK["pixc_pixels"] = npix
C.check(ncell >= 0.99 * t.L * npix, f"only {ncell:,} cells for {npix:,} pixels")
log(f"cells {ncell:,}"); stage("inputs_and_cells", s0)
s0 = time.time(); Pp, Pm, J = C.read_lines(t, K, RR)
nbad = int((~np.isfinite(Pp) | ~np.isfinite(Pm) | ~np.isfinite(J)).sum()); CHK["slc_nonfinite_cells"] = nbad
C.check(nbad <= 0.01 * ncell, f"{nbad:,} SLC cells are not finite (more than 1 %): damaged SLC file?"); stage("read_slc", s0)

# A: rare powers
rp, cnt = C.rare_mean(Pp, PID, npix); rm, _ = C.rare_mean(Pm, PID, npix); full = (cnt == t.L) & (t.pp > 0) & (t.pm > 0)
C.check(full.sum() > 1000, f"only {int(full.sum())} PIXC pixels with all {t.L} SLC lines and valid powers")
ratio_p = t.pp[full] / rp[full]; ratio_m = t.pm[full] / rm[full]
CHK["A_rare_power_ratio_pixc_over_ours_p1_p50_p99"] = dict(plus=pct(ratio_p, [1, 50, 99]), minus=pct(ratio_m, [1, 50, 99]), pixels=int(full.sum()))
log("A rare power ratio PIXC/ours", CHK["A_rare_power_ratio_pixc_over_ours_p1_p50_p99"])
for nm, r_ in (("plus", ratio_p), ("minus", ratio_m)):
    q1, q50, q99 = pct(r_, [1, 50, 99])
    C.check(abs(q50 - 1) < 1e-4 and q1 > 0.999 and q99 < 1.001, f"A failed ({nm}): SLC lines do not reproduce the PIXC rare powers {q1, q50, q99}; wrong SLC/PIXC pair or line offset")
del rp, rm, ratio_p, ratio_m

# ---------------------------------------------------------------- water level, placement and geometry checks
s0 = time.time()
use = t.okp & np.isin(t.cls, (3, 4)) & (np.abs(t.hp) < 1e4); C.check(use.sum() > 1000, "fewer than 1000 PIXC water heights (classes 3-4)")
WLG = C.level_grid(t.px, t.py, t.hp, use, rmax=1000.0, k=64, cell=100.0); wl_pix = np.where(t.okp, C.grid_bilinear(WLG, np.nan_to_num(t.px), np.nan_to_num(t.py)), np.nan)
WLc = wl_pix[PID]; stage("water_level_grid", s0)
# B: geolocation of rare centre lines at the PIXC height vs the PIXC position (random sample)
rng = np.random.default_rng(0); cen = np.where((D == 0) & t.okp[PID] & (np.abs(t.hp[PID]) < 1e4))[0]
C.check(len(cen) > 1000, "fewer than 1000 PIXC pixels with a position and height"); cen = rng.choice(cen, min(200_000, len(cen)), replace=False)
pc_ = PID[cen]; la, lo, _ = C.geoloc(t, K[cen], RR[cen], t.hp[pc_], t.media_slant[pc_])
Sj, _, cl = C.frame(t, K[cen]); far = C.dot(C.geod2ecef(t.plat[pc_], t.plon[pc_], t.hp[pc_]) - Sj, cl) < 0   # PIXC put the point beyond nadir; a height alone cannot tell the side
nor = ~np.isfinite(la); use = ~nor & ~far; x_, y_ = t.to_utm.transform(lo[use], la[use])
dpos = np.hypot(np.asarray(x_) - t.px[pc_[use]], np.asarray(y_) - t.py[pc_[use]]); b50, b90, b99 = pct(dpos, [50, 90, 99])
CHK["B_position"] = dict(error_m_p50_p90_p99=[b50, b90, b99], sampled=int(len(cen)), compared=int(use.sum()), no_root=int(nor.sum()), pixc_beyond_nadir=int(far.sum()))
log("B position", CHK["B_position"]); C.check(use.mean() > 0.9, f"B failed: only {use.mean():.1%} of the sampled pixels could be compared")
C.check(b50 < 2.0 and b99 < 5.0, f"B failed: position error median {b50:.2f} m, p99 {b99:.2f} m (limits 2 m, 5 m)")
del Sj, cl, far, nor, use, dpos
# placement of every cell at the water level (+ the water-level interferometric phase, needed for the coherent power inside the prior water mask)
s0 = time.time(); LAT = np.full(ncell, np.nan); LON = np.full(ncell, np.nan); PHW = np.full(ncell, np.nan, np.float32)
okw = np.where(np.isfinite(WLc))[0]
def place(i0):                                                   # threads write disjoint slices; numpy releases the GIL inside the array maths
    ii = okw[i0:i0 + args.chunk]; la, lo, X = C.geoloc(t, K[ii], RR[ii], WLc[ii], t.media_slant[PID[ii]]); LAT[ii] = la; LON[ii] = lo; PHW[ii] = C.geo_phase(t, X, K[ii])
with ThreadPoolExecutor(max_workers=max(args.threads, 1)) as ex: list(ex.map(place, range(0, len(okw), args.chunk)))
placed = np.isfinite(LAT); CHK["placement"] = dict(cells=ncell, placed=int(placed.sum()), no_water_level=int((~np.isfinite(WLc)).sum()),
                                                  no_root_at_water_level=int((np.isfinite(WLc) & ~placed).sum()), threads=int(args.threads))
log("placement at the water level", CHK["placement"]); gate(placed.mean() > 0.5, f"only {placed.mean():.1%} of the cells could be placed at the water level")
stage("placement_at_water_level", s0)
s0 = time.time(); PHG = C.phi_ref_lines(t, K, RR); CHK["grdem_reference_phase_finite_share"] = float(np.isfinite(PHG).mean())
C.check(np.isfinite(PHG).mean() > 0.9, f"GrDEM reference phase missing for {1 - np.isfinite(PHG).mean():.1%} of the cells (cells outside the GrDEM cross-section)")
stage("grdem_reference_phase", s0)
# C: PIXC interferogram phase = our rare interferogram + GrDEM reference phase (+ one constant)
Jr, _ = C.rare_mean(J, PID, npix); ccen = np.where(D == 0)[0]; phg_pix = np.full(npix, np.nan); phg_pix[PID[ccen]] = PHG[ccen]
psi = C.wrap(np.angle(t.ifg * np.conj(Jr)) - phg_pix); okc = full & np.isfinite(psi) & (np.abs(Jr) > 0)
C.check(okc.sum() > 1000, "fewer than 1000 pixels for the phase check")
c0 = float(np.angle(np.mean(np.exp(1j * psi[okc])))); res = np.abs(C.wrap(psi[okc] - c0))
CHK["C_phase_residual_rad"] = dict(constant=c0, median_abs=float(np.median(res)), p90_abs=float(np.percentile(res, 90)), pixels=int(okc.sum()))
log("C phase residual", CHK["C_phase_residual_rad"]); C.check(np.median(res) < 0.05, f"C failed: median phase residual {np.median(res):.3f} rad")
del Jr, psi, res

# ---------------------------------------------------------------- coherent power (x2_two_knobs.py, reference G)
s0 = time.time(); CJ = np.zeros((t.NA, t.NRg), np.uint8); CJ[t.ai, t.ri] = t.cls; WR = np.isin(CJ, C.WATER)
NB_ALL = ndimage.binary_erosion(WR, np.ones((3, 3), bool), border_value=0); NB_ANY = ndimage.binary_dilation(WR, np.ones((3, 3), bool))
C0 = t.cls[PID]; XPH = np.nan_to_num(t.xo[PID] / t.dpp[PID], nan=0.0, posinf=0.0, neginf=0.0)
MASKW = (t.prior[PID] >= 0.5) & np.isfinite(PHW) & np.isfinite(PHG)          # inside the prior water mask and with a water-level prediction
PRED = np.where(MASKW, C.wrap(PHW - PHG - XPH + np.pi), np.pi - XPH); del PHW, PHG
Pw = 0.5 * (Pp + Pm); IWC = NB_ALL[A, RR] & (C0 == 4); ILC = ~NB_ANY[A, RR] & (C0 == 1)
zz = np.exp(1j * (np.angle(J[IWC]) - PRED[IWC])); zz = zz[np.isfinite(zz)]; C.check(zz.size > 1000 and ILC.sum() > 1000, "too few interior water / land cells for check D")
m_ = np.mean(zz); OFFS = float(np.angle(m_))
PCOH = (Pw + np.abs(J) * np.cos(np.angle(J) - PRED - OFFS)).astype(np.float32); del PRED
CHK["D_coherent_power"] = dict(offset_rad=OFFS, resultant_interior_water=float(np.abs(m_)), gain_water=float(np.nanmean(PCOH[IWC]) / np.nanmean(Pw[IWC])),
                               gain_land=float(np.nanmean(PCOH[ILC]) / np.nanmean(Pw[ILC])), contrast_incoherent=float(np.nanmean(Pw[IWC]) / np.nanmean(Pw[ILC])),
                               contrast_coherent=float(np.nanmean(PCOH[IWC]) / np.nanmean(PCOH[ILC])), prior_mask_share=float(MASKW.mean()))
log("D coherent power", CHK["D_coherent_power"])
gate(np.abs(m_) > 0.2 and CHK["D_coherent_power"]["gain_water"] > CHK["D_coherent_power"]["gain_land"],
     f"D failed: interior water does not add up coherently (resultant {np.abs(m_):.2f}); a phase-prediction error, or very low water coherence")
stage("coherent_power", s0)

# ---------------------------------------------------------------- multilooks
MIN_LINES = 4                                            # densified: fill a cell only when >= 4 of the 7 window lines have values (user rule, 10-13)
s0 = time.time(); F2, cnt2 = C.rare_mean(PCOH, PID, npix); F5, n5 = C.sliding_mean(PCOH, K, RR, t.NL, t.NP, t.L, min_lines=MIN_LINES)
fill5 = n5 >= MIN_LINES; CHK["densified_cells_not_filled"] = int((~fill5).sum())
e = (D == 0) & (cnt2[PID] == t.L) & (n5 == t.L) & np.isfinite(F5) & (F2[PID] > 0); C.check(e.sum() > 1000, "too few full centre lines for check E")
rr_ = F5[e] / F2[PID[e]]; CHK["E_sliding_over_rare_centre_lines_p1_p50_p99"] = pct(rr_, [1, 50, 99])
log("E sliding / rare at centre lines", CHK["E_sliding_over_rare_centre_lines_p1_p50_p99"])
C.check(np.all(np.abs(np.percentile(rr_, [1, 99]) - 1) < 1e-5), "E failed: sliding and rare multilooks differ on centre lines")
stage("multilooks", s0)

# ---------------------------------------------------------------- detection
s0 = time.time(); pri = np.clip(t.prior, 0.05, 0.95); CH = 100; CHK["class_source"] = args.classes; CHK["densify"] = args.densify
cls5 = np.zeros(ncell, np.uint8); wf5 = np.full(ncell, np.nan, np.float32)        # class 0 = not filled (fewer than MIN_LINES lines in the window)
if args.classes == "ours":                                # our detector (gamma likelihood + Ising MRF) on the new values
    cls2, wf2, st2 = C.detect(F2, (t.NA, t.NRg), t.ai, t.ri, t.cls, pri, NB_ALL[t.ai, t.ri] & (t.cls == 4), ~NB_ANY[t.ai, t.ri] & (t.cls == 1),
                              (t.ai // CH) * 100000 + t.ri // 400, 20, t.wfj)
    log("reproduced detection", st2); stage("detect_reproduced", s0)
    s0 = time.time(); f_ = np.where(fill5)[0]
    c_, w_, st5 = C.detect(F5[f_], (t.NL, t.NP), K[f_], RR[f_], C0[f_], pri[PID[f_]], IWC[f_], ILC[f_], (A[f_] // CH) * 100000 + RR[f_] // 400, 140, t.wfj[PID[f_]])
    cls5[f_] = c_; wf5[f_] = w_; del c_, w_, f_
    log("densified detection", st5); stage("detect_densified", s0)
    CHK["F_detection"] = dict(reproduced=st2, densified=st5)
    for nm, st in (("reproduced", st2), ("densified", st5)): gate(st["agree_with_pixc_water_land"] >= 0.8, f"F failed ({nm}): agreement with PIXC water/land {st['agree_with_pixc_water_land']:.3f}")
else:                                                     # the PIXC (JPL) classes, copied: the 7 lines of a pixel all take its class
    cls2 = t.cls.copy(); wf2 = t.wfj.astype(np.float32); cls5[fill5] = C0[fill5]; wf5[fill5] = t.wfj[PID[fill5]]
    CHK["F_detection"] = "skipped: classes copied from the PIXC (--classes pixc)"; log("classes copied from the PIXC; check F skipped"); stage("classes_from_pixc", s0)

if args.densify == "landstop":                            # user rule 10-08 (x22 5Gs): the moving average stops at land lines
    s0 = time.time(); G1 = np.zeros((t.NL, t.NP), np.uint8)
    G1[K, RR] = cls5 if args.classes == "ours" else C0      # pass 1: classes of the plain moving average (ours) or the PIXC classes
    barrier = C.isolated_water_to_land(G1) == 1; del G1
    F5, n5, isb5 = C.landstop_mean(PCOH, K, RR, t.NL, t.NP, t.L, barrier, min_lines=MIN_LINES); del barrier
    fill5 = np.isfinite(F5); cls5 = np.zeros(ncell, np.uint8); wf5 = np.full(ncell, np.nan, np.float32)
    CHK["landstop"] = dict(land_line_cells=int(isb5.sum()), window_lines_p5_p50_mean=[float(np.percentile(n5[~isb5], 5)), float(np.median(n5[~isb5])), float(n5[~isb5].mean())],
                           not_filled=int((~fill5).sum()))
    if args.classes == "ours":                            # pass 2: our detector on the new values
        f_ = np.where(fill5)[0]
        c_, w_, st5 = C.detect(F5[f_], (t.NL, t.NP), K[f_], RR[f_], C0[f_], pri[PID[f_]], IWC[f_], ILC[f_], (A[f_] // CH) * 100000 + RR[f_] // 400, 140, t.wfj[PID[f_]])
        cls5[f_] = c_; wf5[f_] = w_; del c_, w_, f_; CHK["F_detection"]["densified_landstop"] = st5
        gate(st5["agree_with_pixc_water_land"] >= 0.8, f"F failed (densified, landstop): agreement with PIXC water/land {st5['agree_with_pixc_water_land']:.3f}")
    else:
        cls5[fill5] = C0[fill5]; wf5[fill5] = t.wfj[PID[fill5]]
    CHK["densified_cells_not_filled"] = int((~fill5).sum()); log("landstop moving average", CHK["landstop"]); stage("densify_landstop", s0)

# ---------------------------------------------------------------- RiverScope matching and cell-level agreement
RSC = np.full(ncell, -1, np.int8); RSP = np.full(npix, -1, np.int8)
if RS_RECS is not None:
    s0 = time.time(); RSC = C.riverscope_sample(LAT, LON, RS_RECS); RSP = C.riverscope_sample(t.plat, t.plon, RS_RECS)
    C.check(set(np.unique(RSC)) <= {-1, 0, 1}, "RiverScope labels must be -1/0/1")
    CHK["riverscope_cells_labelled"] = int((RSC >= 0).sum())
    def agree(water, rs, w=None):
        w = np.ones(len(rs)) if w is None else np.nan_to_num(w); ok = rs >= 0; rw = ok & (rs == 1); rl = ok & (rs == 0); dw, dl = w[rw].sum(), w[ok & water].sum()
        return dict(missed_pct=float(100 * w[rw & ~water].sum() / dw) if dw > 0 else float("nan"),
                    false_pct=float(100 * w[rl & water].sum() / dl) if dl > 0 else float("nan"), n=int(ok.sum()))
    CHK["cell_agreement_with_riverscope"] = {"existing_on_cells": agree(np.isin(C0, C.WATER), RSC), "reproduced_block_on_cells": agree(np.isin(cls2[PID], C.WATER), RSC),
                                             "densified_on_cells": agree(np.isin(cls5, C.WATER), RSC),
                                             "existing_at_pixc_positions_area_weighted": agree(np.isin(t.cls, C.WATER), RSP, t.area),
                                             "reproduced_at_pixc_positions_area_weighted": agree(np.isin(cls2, C.WATER), RSP, t.area)}
    log("cell agreement with RiverScope", CHK["cell_agreement_with_riverscope"]); stage("riverscope", s0)

# ---------------------------------------------------------------- write (to a .part file, renamed at the end: a stopped job leaves no half file)
s0 = time.time(); import netCDF4 as nc
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True); tmp = args.out + ".part"; o = nc.Dataset(tmp, "w", format="NETCDF4")
o.title = "densepixc dense pixel cloud"; o.slc_file = os.path.basename(args.slc); o.pixc_file = os.path.basename(args.pixc); o.tile_name = t.tile_name
o.num_azimuth_looks = t.L; o.azimuth_offset = t.OFF; o.utm_epsg = t.epsg; o.coherent_power_offset_rad = OFFS; o.water_classes = "3 4 5 6 7"
o.class_source = args.classes; o.min_window_lines_densified = MIN_LINES; o.densify = args.densify
o.placement = "cell centres at the water level: median of the 64 nearest PIXC class 3-4 heights within 1 km, 100 m grid, bilinear; NaN = could not be placed"
gc = o.createGroup("cells"); gc.createDimension("cell", ncell)
def var(g, name, arr, dtype, units="", desc="", classes=False):
    v = g.createVariable(name, dtype, (list(g.dimensions)[0],), zlib=True, complevel=2, fill_value=(np.nan if dtype.startswith("f") else None))
    v[:] = arr; v.units = units; v.long_name = desc
    if classes:
        for k_, val in t.class_flags.items(): v.setncattr(k_, val)
        v.comment = "PIXC classification codes; 0 = no class (PIXC fill)"
var(gc, "slc_line", K, "i4", "", "SLC line index (0-based, num_lines dimension of the SLC file)")
var(gc, "range_index", RR, "i2", "", "range sample (= PIXC range_index)"); var(gc, "pixc_index", PID, "i4", "", "row of the parent pixel in the PIXC pixel_cloud arrays")
var(gc, "line_offset", D, "i1", "", "SLC line minus the centre line of the parent PIXC pixel (-3..3)")
var(gc, "latitude", LAT, "f8", "degrees_north", "cell centre at the water level"); var(gc, "longitude", LON, "f8", "degrees_east", "cell centre at the water level")
var(gc, "water_level", WLc, "f4", "m", "water level used for placement (ellipsoid height)")
var(gc, "power_db", 10 * np.log10(np.maximum(Pw, 1e-30)), "f4", "dB", "per-line mean of the two antenna powers")
var(gc, "coherent_power", PCOH, "f4", "", "per-line coherent power (P+ + P-)/2 + |z+ z-*| cos(phase - prediction - offset)")
var(gc, "coherent_power_reproduced", F2[PID], "f4", "", "7-line multilook of the parent pixel, on all its lines (block-filled)")
var(gc, "coherent_power_densified", F5, "f4", "", "sliding 7-line multilook (moving average along azimuth; see attribute densify)")
var(gc, "densified_window_lines", n5, "i2", "", "number of lines averaged for coherent_power_densified")
var(gc, "class_existing", C0, "u1", "", "PIXC classification of the parent pixel", classes=True)
var(gc, "class_reproduced", cls2[PID], "u1", "", "our class of the 7-line multilook, on all lines of the pixel (block-filled)", classes=True)
var(gc, "class_densified", cls5, "u1", "", "our class of the sliding 7-line multilook; 0 = not filled (fewer than 4 of 7 window lines)", classes=True)
var(gc, "water_frac_reproduced", wf2[PID], "f4", "1", "water fraction of the 7-line multilook"); var(gc, "water_frac_densified", wf5, "f4", "1", "water fraction of the sliding multilook")
var(gc, "riverscope", RSC, "i1", "", "RiverScope label at the cell centre: 1 water, 0 land, -1 none")
if not args.no_ifg:
    var(gc, "power_plus", Pp, "f4", "", "|z+|^2"); var(gc, "power_minus", Pm, "f4", "", "|z-|^2")
    var(gc, "ifg_real", J.real.astype(np.float32), "f4", "", "Re z+ conj(z-)"); var(gc, "ifg_imag", J.imag.astype(np.float32), "f4", "", "Im z+ conj(z-)")
gp = o.createGroup("pixc"); gp.createDimension("points", npix)
var(gp, "latitude", t.plat, "f8", "degrees_north", "PIXC latitude"); var(gp, "longitude", t.plon, "f8", "degrees_east", "PIXC longitude")
var(gp, "pixel_area", t.area, "f4", "m2", "PIXC pixel_area"); var(gp, "azimuth_index", t.ai, "i4"); var(gp, "range_index", t.ri, "i2")
var(gp, "class_existing", t.cls, "u1", "", "PIXC classification", classes=True); var(gp, "class_reproduced", cls2, "u1", "", "our class of the 7-line multilook", classes=True)
var(gp, "coherent_power_reproduced", F2, "f4", "", "7-line multilook"); var(gp, "water_frac_reproduced", wf2, "f4", "1", "water fraction of the 7-line multilook")
var(gp, "riverscope", RSP, "i1", "", "RiverScope label at the PIXC position: 1 water, 0 land, -1 none")
# the attribute holds check values only; status, times and memory go to the JSON file
o.checks_json = json.dumps({k: v for k, v in CHK.items() if k not in ("status", "timing_s", "peak_memory_gib")}, default=float)
o.close(); os.replace(tmp, args.out); stage("write", s0)
CHK["warnings"] = WARN; dump("passed" if not WARN else "passed_with_warnings")
log("all checks passed" if not WARN else f"passed with {len(WARN)} warnings", "; wrote", args.out, "and", JPATH, "| peak memory GiB", CHK["peak_memory_gib"])
