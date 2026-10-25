"""densepixc core: SWOT L1B_HR_SLC + L2_HR_PIXC (same tile) -> reproduced and densified pixel cloud (water/land), matched to PIXC and RiverScope.

The algorithms are ported from scripts of the slc_to_pixc study that were checked against the real PIXC (the study's README.md):
  rare powers / interferogram ........ r1_rare_power.py   (PIXC power_plus_y / ours: 1 +- 2e-7, 1st-99th percentile, tile 017_220_037L)
  GrDEM reference phase .............. r2b_ref_phase_lines.py, x2_two_knobs.py:phi_ref_lines (phase residual median 0.0087 rad)
  geolocation ........................ r3d_geoloc_nofit.py, x2_two_knobs.py:geoloc_native (position median 0.21 m without, 0.006 m with the 1.23 cm range term)
  coherent power, gamma MAP + Ising MRF detection, water-level grid ........ x2_two_knobs.py (run_product, icm, level_grid)
Heavy dependencies (netCDF4, pyproj, rasterio) are imported inside the functions that need them, so the pure-array functions can be unit tested anywhere."""
import warnings
import numpy as np
from scipy import ndimage
from scipy.special import gammaln

WATER = (3, 4, 5, 6, 7)          # PIXC classes counted as water (class 2 'land near water' is land)
A_ = 6378137.0; F_ = 1 / 298.257223563; E2 = F_ * (2 - F_)
RANGE_BIAS_M = 0.0123             # r3d_geoloc_nofit.py: best range convention 'SLC near_range + media/cos(inc) + 1.23 cm'


class CheckError(AssertionError):
    """raised when a built-in consistency check fails (input pairing, indexing, geometry, product sanity)"""


def check(cond, msg):
    if not bool(cond):
        raise CheckError(msg)


# ---------------------------------------------------------------- geometry primitives
def geod2ecef(lat, lon, h):
    p_, l_ = np.radians(lat), np.radians(lon); N = A_ / np.sqrt(1 - E2 * np.sin(p_) ** 2)
    return np.c_[(N + h) * np.cos(p_) * np.cos(l_), (N + h) * np.cos(p_) * np.sin(l_), (N * (1 - E2) + h) * np.sin(p_)]


def ecef2geod(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]; lon = np.arctan2(y, x); pp = np.hypot(x, y); lat = np.arctan2(z, pp * (1 - E2))
    for _ in range(5):
        N = A_ / np.sqrt(1 - E2 * np.sin(lat) ** 2); h = pp / np.cos(lat) - N; lat = np.arctan2(z, pp * (1 - E2 * N / (N + h)))
    N = A_ / np.sqrt(1 - E2 * np.sin(lat) ** 2); h = pp / np.cos(lat) - N
    return np.degrees(lat), np.degrees(lon), h


def dot(a, b):
    return np.einsum("ij,ij->i", a, b)


def wrap(x):
    return np.angle(np.exp(1j * x))


def at(arr, jf):
    """linear interpolation of a per-record array (n, 3) at fractional record index jf"""
    jf = np.clip(jf, 0, len(arr) - 1.000001); i = np.floor(jf).astype(np.int64); w = (jf - i)[:, None]
    return (1 - w) * arr[i] + w * arr[i + 1]


# ---------------------------------------------------------------- inputs
class Tile:
    """everything the pipeline needs from one SLC file and its PIXC file"""


def load_tile(slc_path, pixc_path):
    import netCDF4 as nc
    t = Tile(); t.slc_path, t.pixc_path = slc_path, pixc_path
    p = nc.Dataset(pixc_path); p.set_auto_mask(False); pc = p["pixel_cloud"]
    s = nc.Dataset(slc_path); s.set_auto_mask(False)
    # pairing checks: same cycle / pass / tile and the same SLC line numbering
    for att in ("cycle_number", "pass_number", "tile_number", "swath_side"):
        if att in p.ncattrs() and att in s.ncattrs():
            check(str(p.getncattr(att)) == str(s.getncattr(att)), f"SLC and PIXC differ in {att}: {s.getncattr(att)} vs {p.getncattr(att)}")
    check(int(p.slc_first_line_index_in_tvp) == int(s.slc_first_line_index_in_tvp), "slc_first_line_index_in_tvp differs between PIXC and SLC")
    t.tile_name = str(p.getncattr("tile_name")) if "tile_name" in p.ncattrs() else ""
    t.side = 1.0 if str(p.getncattr("swath_side") if "swath_side" in p.ncattrs() else t.tile_name[-1:]).upper().startswith("L") else -1.0
    t.L = int(round(float(pc.num_azimuth_looks))); t.OFF = int(pc.azimuth_offset); t.NA = int(pc.interferogram_size_azimuth); t.NRg = int(pc.interferogram_size_range)
    # azimuth_offset is 3-9 on the study's tiles (3 + (1 - record_counter) mod 7); check A tests the line mapping itself
    check(t.L == 7 and 0 <= t.OFF < 3 * t.L, f"unexpected num_azimuth_looks / azimuth_offset: {t.L} / {t.OFF}")
    t.SF = int(p.slc_first_line_index_in_tvp); t.DR = float(p.nominal_slant_range_spacing); t.LAM = float(p.wavelength)
    g = s["slc"]; t.NL, t.NP = g["slc_plus_y"].shape[:2]; t.NR_SLC = float(s.near_range)
    check(t.NP == t.NRg, f"SLC num_pixels {t.NP} != PIXC interferogram_size_range {t.NRg}")
    check(t.NA * t.L <= t.NL + t.L, f"PIXC has {t.NA} rare lines but the SLC only {t.NL} lines")
    t.fill_counts = {}
    def rd(name):                                                  # float variable; _FillValue and non-physical values (|x| >= 1e30) -> NaN
        var = pc[name]; a = np.array(var[:], dtype=float); fv = getattr(var, "_FillValue", None); bad = ~(np.abs(a) < 1e30)
        if fv is not None: bad |= a == float(np.asarray(fv, dtype=var.dtype))
        t.fill_counts[name] = int(bad.sum()); a[bad] = np.nan; return a
    v = {k: rd(k) for k in ("latitude", "longitude", "height", "height_cor_xover", "dheight_dphase", "model_dry_tropo_cor", "model_wet_tropo_cor",
                            "iono_cor_gim_ka", "inc", "prior_water_prob", "water_frac", "power_plus_y", "power_minus_y", "pixel_area")}
    t.ai = pc["azimuth_index"][:].astype(np.int64); t.ri = pc["range_index"][:].astype(np.int64); cls = pc["classification"][:].astype(np.int64)
    badc = ~np.isin(cls, (1, 2, 3, 4, 5, 6, 7)); t.fill_counts["classification"] = int(badc.sum()); cls[badc] = 0      # 0 = not a class: never decided, never water
    t.cls = cls.astype(np.uint8); cv = pc["classification"]
    t.class_flags = {a: cv.getncattr(a) for a in ("flag_values", "flag_meanings") if a in cv.ncattrs()}   # copied to the output class variables
    check(t.ai.min() >= 0 and t.ai.max() < t.NA and t.ri.min() >= 0 and t.ri.max() < t.NRg, "PIXC azimuth/range index out of the interferogram size")
    check(len(np.unique(t.ai * t.NRg + t.ri)) == len(t.ai), "duplicate (azimuth_index, range_index) pairs in the PIXC")
    t.plat = v["latitude"].astype(float); t.plon = v["longitude"].astype(float); t.hp = v["height"].astype(float)
    t.xo = v["height_cor_xover"].astype(float); t.dpp = v["dheight_dphase"].astype(float); t.inc = np.radians(v["inc"].astype(float))   # PIXC inc is in degrees
    t.media = (v["model_dry_tropo_cor"] + v["model_wet_tropo_cor"] + v["iono_cor_gim_ka"]).astype(float)
    okm = (np.abs(t.media) < 10) & (t.inc > 0) & (t.inc < 0.2); check(okm.mean() > 0.9, "media corrections or incidence angles missing for >10 % of PIXC pixels")
    t.MEDIA = float(np.median(t.media[okm] / np.cos(t.inc[okm])))
    t.media_slant = np.where(okm, t.media / np.cos(t.inc), t.MEDIA)                     # per pixel slant-range media correction (r3d convention)
    pv = v["prior_water_prob"].astype(float); t.prior = np.where((pv >= 0) & (pv <= 1), pv, 0.5)
    t.wfj = np.clip(np.nan_to_num(v["water_frac"].astype(float), nan=0.0), 0, 1)
    t.pp = v["power_plus_y"]; t.pm = v["power_minus_y"]; t.area = v["pixel_area"]
    ig = np.array(pc["interferogram"][:], dtype=float); badi = ~(np.abs(ig) < 1e30).all(axis=1); t.fill_counts["interferogram"] = int(badi.sum())
    t.ifg = ig[:, 0] + 1j * ig[:, 1]; t.ifg[badi] = np.nan
    t.okp = (np.abs(t.plat) <= 90) & (np.abs(t.plon) <= 360)
    tv = s["tvp"]; t.T = tv["time"][:]; t.S = np.c_[tv["x"][:], tv["y"][:], tv["z"][:]]; t.V = np.c_[tv["vx"][:], tv["vy"][:], tv["vz"][:]]
    t.Ap = np.c_[tv["plus_y_antenna_x"][:], tv["plus_y_antenna_y"][:], tv["plus_y_antenna_z"][:]]
    t.Am = np.c_[tv["minus_y_antenna_x"][:], tv["minus_y_antenna_y"][:], tv["minus_y_antenna_z"][:]]
    check(t.SF + t.NL <= len(t.S), "the SLC lines run past the end of the TVP records")
    gd = s["grdem"]; t.GH = gd["height"][:].astype(float); t.GT = gd["platform_time"][:]; t.GLA = gd["platform_latitude"][:]; t.GLO = gd["platform_longitude"][:]
    t.GV = np.c_[gd["platform_velocity_x"][:], gd["platform_velocity_y"][:], gd["platform_velocity_z"][:]]
    t.SCT = float(gd.grdem_min_cross_track) + float(gd.grdem_cross_track_spacing) * np.arange(t.GH.shape[1])
    t._slc = s; t._pixc = p
    import pyproj
    lon_c = float(np.median(t.plon[t.okp])); t.epsg = 32600 + int(((lon_c + 180) % 360) // 6) + 1 + (0 if np.median(t.plat[t.okp]) >= 0 else 100)
    t.to_utm = pyproj.Transformer.from_crs(4326, t.epsg, always_xy=True); t.to_geo = pyproj.Transformer.from_crs(t.epsg, 4326, always_xy=True)
    t.px = np.full(len(t.ai), np.nan); t.py = np.full(len(t.ai), np.nan); x_, y_ = t.to_utm.transform(t.plon[t.okp], t.plat[t.okp]); t.px[t.okp] = x_; t.py[t.okp] = y_
    return t


# ---------------------------------------------------------------- the dense cell set and SLC reading
def dense_cells(ai, ri, L, OFF, NL):
    """every SLC line of every saved PIXC pixel: rare line a covers SLC lines a*L + OFF + d, d = -(L//2) .. L//2 (r1: centre line 7a+4 for OFF = 4).
    Returns K (SLC line), RR (range sample), PID (row in the PIXC arrays), D (line offset d), sorted by K then RR."""
    d = np.arange(-(L // 2), L // 2 + 1); n = len(ai)
    K = (np.repeat(ai * L + OFF, len(d)) + np.tile(d, n)).astype(np.int64); PID = np.repeat(np.arange(n), len(d)); D = np.tile(d, n).astype(np.int8)
    ok = (K >= 0) & (K < NL); K, PID, D = K[ok], PID[ok], D[ok]; RR = ri[PID]
    o = np.lexsort((RR, K)); return K[o], RR[o].astype(np.int64), PID[o].astype(np.int64), D[o]


def read_lines(t, K, RR, chunk=1500, log=print):
    """per cell: |z+|^2, |z-|^2 and z+ conj(z-) from the SLC (K sorted ascending)"""
    check(np.all(np.diff(K) >= 0), "cells must be sorted by SLC line")
    g = t._slc["slc"]; n = len(K); Pp = np.zeros(n, np.float32); Pm = np.zeros(n, np.float32); J = np.zeros(n, np.complex64)
    for k0 in range(int(K.min()), int(K.max()) + 1, chunk):
        k1 = min(k0 + chunk, t.NL); a_, b_ = np.searchsorted(K, [k0, k1])
        if b_ == a_: continue
        zp = g["slc_plus_y"][k0:k1].astype(np.float32); zm = g["slc_minus_y"][k0:k1].astype(np.float32); kk = K[a_:b_] - k0; rr = RR[a_:b_]
        zpc = zp[kk, rr, 0] + 1j * zp[kk, rr, 1]; zmc = zm[kk, rr, 0] + 1j * zm[kk, rr, 1]; del zp, zm
        Pp[a_:b_] = np.abs(zpc) ** 2; Pm[a_:b_] = np.abs(zmc) ** 2; J[a_:b_] = zpc * np.conj(zmc)
    return Pp, Pm, J


def rare_mean(values, PID, npix):
    """mean over the SLC lines of each PIXC pixel (the 'rare' 7-line multilook); complex values allowed. Non-finite values are left out;
    cnt = number of finite lines per pixel"""
    v = np.asarray(values); fin = np.isfinite(v); cnt = np.bincount(PID, weights=fin.astype(float), minlength=npix)
    if np.iscomplexobj(v):
        s_ = np.bincount(PID, weights=np.where(fin, v.real, 0.0).astype(float), minlength=npix) + 1j * np.bincount(PID, weights=np.where(fin, v.imag, 0.0).astype(float), minlength=npix)
    else:
        s_ = np.bincount(PID, weights=np.where(fin, v, 0.0).astype(float), minlength=npix)
    out = np.full(npix, np.nan, dtype=complex if np.iscomplexobj(v) else float); ok = cnt > 0; out[ok] = s_[ok] / cnt[ok]; return out, cnt


def sliding_mean(values, K, RR, NL, NP, L, min_lines=1):
    """mean over the existing cells in the L-line window centred on each cell (same range sample): the sliding multilook (no decimation).
    A cell whose window holds fewer than min_lines finite values gets NaN (not filled). n = finite values in the window."""
    fin = np.isfinite(values); Gs = np.zeros((NL, NP), np.float32); Gn = np.zeros((NL, NP), np.float32)
    Gs[K, RR] = np.where(fin, values, 0.0); Gn[K, RR] = fin                    # a non-finite cell is left out (a running sum would carry it down the column)
    Gs = ndimage.uniform_filter1d(Gs, L, axis=0, mode="constant"); Gn = ndimage.uniform_filter1d(Gn, L, axis=0, mode="constant")
    out = (Gs[K, RR] / np.maximum(Gn[K, RR], 1e-6)).astype(np.float64); n = np.rint(Gn[K, RR] * L).astype(np.int16)
    out[n < max(min_lines, 1)] = np.nan                                        # too few lines in the window: no value (never 0)
    return out, n


# ---------------------------------------------------------------- moving average that stops at land (user rule 10-08, x22_box.py)
def isolated_water_to_land(G):
    """G: class grid (SLC lines x range samples, 0 = no cell). Water components (classes 3-7, 8-connected) with no class 4 or 5 cell that do not touch
    the grid edge become land (1), and so do class-2 cells that touch only such components (x22_box.py isolated_to_land, user rule 10-08)."""
    SQ8 = np.ones((3, 3), bool); W = np.isin(G, WATER); lab, n = ndimage.label(W, SQ8); out = G.copy()
    if n == 0: return out
    core = np.asarray(ndimage.maximum(np.isin(G, (4, 5)).astype(np.int8), lab, np.arange(1, n + 1))); edge = np.zeros(n + 1, bool)
    for sl in (lab[0], lab[-1], lab[:, 0], lab[:, -1]): edge[np.unique(sl)] = True
    bad = np.zeros(n + 1, bool); bad[1:] = (core == 0) & ~edge[1:]; drop = bad[lab]; del lab
    drop2 = (G == 2) & ndimage.binary_dilation(drop, SQ8) & ~ndimage.binary_dilation(W & ~drop, SQ8)
    out[drop | drop2] = 1; return out
def landstop_mean(values, K, RR, NL, NP, L, barrier, min_lines=1):
    """moving average that does not cross land (user rule 10-08, x22_box.py barrier_rolling with SHIFT): for line k the usable lines are the run of
    non-barrier lines between the nearest barrier (land) lines on each side; the window is L lines of that run (fewer if the run is shorter), as centred
    on k as the run allows. A barrier line keeps its own value. A non-barrier cell whose window holds fewer than min_lines finite values gets NaN.
    Returns the mean, the number of finite values used, and whether the cell is a barrier line."""
    half = L // 2; rows = np.arange(NL, dtype=np.int32)[:, None]; K = np.asarray(K, np.int64); RR = np.asarray(RR, np.int64); isb = barrier[K, RR]
    lb = np.where(barrier, rows, -1).astype(np.int32); np.maximum.accumulate(lb, axis=0, out=lb); s0 = lb[K, RR].astype(np.int64) + 1; del lb
    nb = np.where(barrier, rows, NL).astype(np.int32); nb = np.minimum.accumulate(nb[::-1], axis=0)[::-1]; s1 = nb[K, RR].astype(np.int64) - 1; del nb
    lw = np.minimum(2 * half + 1, s1 - s0 + 1); st_ = np.clip(K - half, s0, np.maximum(s1 - lw + 1, s0))
    lo = np.where(isb, K, st_); hi = np.where(isb, K, st_ + lw - 1)
    fin = np.isfinite(values); V = np.zeros((NL, NP)); V[K, RR] = np.where(fin, values, 0.0); CS = np.zeros((NL + 1, NP)); np.cumsum(V, axis=0, out=CS[1:]); del V
    s_ = CS[hi + 1, RR] - CS[lo, RR]; del CS
    Cn = np.zeros((NL, NP), np.float32); Cn[K, RR] = fin; CN = np.zeros((NL + 1, NP), np.float64); np.cumsum(Cn, axis=0, out=CN[1:]); del Cn
    n = np.rint(CN[hi + 1, RR] - CN[lo, RR]).astype(np.int16); del CN
    out = s_ / np.maximum(n, 1); out[(n < 1) | (~isb & (n < max(min_lines, 1)))] = np.nan
    return out, n, isb


# ---------------------------------------------------------------- water-level grid from PIXC heights (x2_two_knobs.py:level_grid)
class Grid:
    pass


def level_grid(x, y, h, use, rmax=1000.0, k=64, cell=100.0, pad=1000.0):
    """median of the k nearest PIXC heights within rmax on a regular grid; empty nodes filled from the nearest filled node"""
    from scipy.spatial import cKDTree
    gr = Grid(); gr.GL = cell; gr.X0 = np.nanmin(x[use]) - pad; gr.Y0 = np.nanmin(y[use]) - pad
    gxs = np.arange(gr.X0, np.nanmax(x[use]) + pad, cell) + cell / 2; gys = np.arange(gr.Y0, np.nanmax(y[use]) + pad, cell) + cell / 2; LX, LY = np.meshgrid(gxs, gys)
    idx = np.where(use)[0]; kt = cKDTree(np.c_[x[idx], y[idx]]); hh = h[idx]; out = np.full(LX.size, np.nan); q = np.c_[LX.ravel(), LY.ravel()]
    for i0 in range(0, len(q), 200000):
        dd, ii = kt.query(q[i0:i0 + 200000], k=min(k, len(idx)), distance_upper_bound=rmax)
        dd = dd.reshape(len(dd), -1); ii = ii.reshape(len(ii), -1)
        with np.errstate(all="ignore"), warnings.catch_warnings():               # grid nodes with no height within rmax: NaN, filled below
            warnings.simplefilter("ignore", RuntimeWarning); out[i0:i0 + 200000] = np.nanmedian(np.where(np.isfinite(dd), hh[np.minimum(ii, len(hh) - 1)], np.nan), axis=1)
    G = out.reshape(LX.shape); check(np.isfinite(G).any(), "water-level grid is empty (no PIXC water heights)")
    _, (iy_, ix_) = ndimage.distance_transform_edt(~np.isfinite(G), return_indices=True); gr.G = G[iy_, ix_]; return gr


def grid_bilinear(gr, x, y):
    fx = (x - gr.X0) / gr.GL - 0.5; fy = (y - gr.Y0) / gr.GL - 0.5; G = gr.G
    ix = np.clip(np.floor(fx).astype(np.int64), 0, G.shape[1] - 2); iy = np.clip(np.floor(fy).astype(np.int64), 0, G.shape[0] - 2)
    wx = np.clip(fx - ix, 0, 1); wy = np.clip(fy - iy, 0, 1)
    return (1 - wy) * ((1 - wx) * G[iy, ix] + wx * G[iy, ix + 1]) + wy * ((1 - wx) * G[iy + 1, ix] + wx * G[iy + 1, ix + 1])


# ---------------------------------------------------------------- radar geometry (x2_two_knobs.py / r3d_geoloc_nofit.py)
LOOK_MAX_DEG = 6.0                # KaRIn looks are about 0.6-4 degrees (swath 10-60 km from nadir); the bisection searches 0 (nadir) to 6 degrees on the swath side
def frame(t, k):
    """TVP position of record k + SF (k fractional ok), the downward ellipsoid normal and the horizontal cross-track unit vector toward the swath"""
    j = np.asarray(k, float) + t.SF; Sj = at(t.S, j); Vj = at(t.V, j); la_, lo_, _ = ecef2geod(Sj); ph, lm = np.radians(la_), np.radians(lo_)
    n = np.c_[np.cos(ph) * np.cos(lm), np.cos(ph) * np.sin(lm), np.sin(ph)]; vh = Vj - dot(Vj, n)[:, None] * n; vh /= np.linalg.norm(vh, axis=1)[:, None]
    dn = -n; return Sj, dn, np.cross(vh, dn) * t.side
def geoloc(t, k, r, h, media_slant):
    """ground point of SLC line k (fractional ok), range sample r, at ellipsoid height h: range sphere about the TVP position of record k + SF with
    radius near_range + r*dr + media/cos(inc) + 1.23 cm (media = model_dry_tropo_cor + model_wet_tropo_cor + iono_cor_gim_ka, negative numbers),
    in the plane through that position normal to the horizontal velocity (as r3d; not the zero-Doppler plane); bisection on the look angle, 0 to 6 degrees
    (as r3d_geoloc_nofit.py). Where the sphere does not reach height h inside that interval (near-nadir terrain above h, or a point beyond 6 degrees)
    there is no root: latitude, longitude and X are NaN, never a point pinned to the interval end."""
    Sj, dn, cl = frame(t, k); R = t.NR_SLC + t.DR * np.asarray(r, float) + media_slant + RANGE_BIAS_M; h = np.asarray(h, float)
    def point(b): return Sj + R[:, None] * (np.cos(b)[:, None] * dn + np.sin(b)[:, None] * cl)
    lo = np.zeros(len(R)); hi = np.full(len(R), np.radians(LOOK_MAX_DEG))
    ok = (ecef2geod(point(lo))[2] - h <= 0) & (ecef2geod(point(hi))[2] - h >= 0)
    for _ in range(27):                                               # 6 deg / 2**27 = 0.7 mm at 900 km range
        b = (lo + hi) / 2; fz = ecef2geod(point(b))[2] - h; lo = np.where(fz < 0, b, lo); hi = np.where(fz < 0, hi, b)
    X = point((lo + hi) / 2); X[~ok] = np.nan
    with np.errstate(invalid="ignore"): la, lo2, _ = ecef2geod(X)
    return la, lo2, X
def geo_phase(t, X, k):
    j = (np.asarray(k) + t.SF).astype(np.int64); return -(2 * np.pi / t.LAM) * (np.linalg.norm(X - t.Ap[j], axis=1) - np.linalg.norm(X - t.Am[j], axis=1))


def phi_ref_lines(t, k, r):
    """GrDEM flattening phase at SLC line k, range sample r (r2b rule): first intersection of the range circle with the GrDEM cross-section of the
    line's own record (SLC near range, no media correction)"""
    import pyproj
    GEOD = pyproj.Geod(ellps="WGS84"); out = np.full(len(k), np.nan); order = np.argsort(k, kind="stable"); ks = k[order]
    uk, st_ = np.unique(ks, return_index=True); en_ = np.r_[st_[1:], len(ks)]
    for kk, a_, b_ in zip(uk, st_, en_):
        j = int(kk) + t.SF; tt = t.T[j]; gg = np.interp(tt, t.GT, np.arange(len(t.GT), dtype=float)); gi = min(int(gg), len(t.GT) - 2); gw = gg - gi
        hl = (1 - gw) * t.GH[gi] + gw * t.GH[gi + 1]; la0 = (1 - gw) * t.GLA[gi] + gw * t.GLA[gi + 1]; lo0 = (1 - gw) * t.GLO[gi] + gw * t.GLO[gi + 1]; v = (1 - gw) * t.GV[gi] + gw * t.GV[gi + 1]
        ph, lm = np.radians(la0), np.radians(lo0); e_ = np.array([-np.sin(lm), np.cos(lm), 0]); nn = np.array([-np.sin(ph) * np.cos(lm), -np.sin(ph) * np.sin(lm), np.cos(ph)])
        az = np.degrees(np.arctan2(v @ e_, v @ nn)) - 90.0 * t.side
        loc, lac, _ = GEOD.fwd(np.full(len(t.SCT), lo0), np.full(len(t.SCT), la0), np.full(len(t.SCT), az), t.SCT)
        X = geod2ecef(lac, loc, hl); Rc = np.linalg.norm(X - t.S[j], axis=1); Rmax = np.maximum.accumulate(Rc)
        idx = order[a_:b_]; Rr = t.NR_SLC + t.DR * r[idx]; q = np.clip(np.searchsorted(Rmax, Rr), 1, len(t.SCT) - 1)
        w = np.clip((Rr - Rc[q - 1]) / np.where(Rc[q] != Rc[q - 1], Rc[q] - Rc[q - 1], 1.0), 0, 1)[:, None]; Xr = (1 - w) * X[q - 1] + w * X[q]
        inside = (Rr >= Rc[0]) & (Rr <= Rmax[-1])                     # outside the GrDEM cross-section: NaN, not the phase of the end node
        out[idx] = np.where(inside, -(2 * np.pi / t.LAM) * (np.linalg.norm(Xr - t.Ap[j], axis=1) - np.linalg.norm(Xr - t.Am[j], axis=1)), np.nan)
    return out


# ---------------------------------------------------------------- detection (x2_two_knobs.py:nll, icm, run_product; ATBD 3.3)
def nll(F, mu, L):
    """negative log-likelihood of a gamma-distributed power F with mean mu and L looks (constant terms kept)"""
    return gammaln(L) - L * np.log(L / mu) - (L - 1) * np.log(np.maximum(F, 1e-30)) + L * F / mu


def icm(shape, ii, jj, free, lab, Dv, beta=2.0, sweeps=8):
    """Ising MRF by iterated conditional modes on a 4-neighbour grid, checkerboard updates; Dv = log-likelihood ratio water vs land (+ log prior)"""
    G_ = np.full(shape, -1, np.int8); G_[ii, jj] = lab.astype(np.int8); FR = np.zeros(shape, bool); FR[ii[free], jj[free]] = True
    DG = np.zeros(shape, np.float32); DG[ii[free], jj[free]] = Dv[free]
    PAR = ((np.arange(shape[0], dtype=np.int64)[:, None] % 2) ^ (np.arange(shape[1], dtype=np.int64)[None, :] % 2)).astype(np.int8)
    nchg = 0
    for _ in range(sweeps):
        nchg = 0
        for p_ in (0, 1):
            Wm = (G_ == 1).astype(np.int8); Lm = (G_ == 0).astype(np.int8); nwc = np.zeros(shape, np.int8); nlc = np.zeros(shape, np.int8)
            for M_, N_ in ((Wm, nwc), (Lm, nlc)):
                N_[1:] += M_[:-1]; N_[:-1] += M_[1:]; N_[:, 1:] += M_[:, :-1]; N_[:, :-1] += M_[:, 1:]
            upd = FR & (PAR == p_); newv = ((DG + beta * (nwc.astype(np.float32) - nlc.astype(np.float32))) > 0).astype(np.int8)
            nchg += int((upd & (newv != G_)).sum()); G_[upd] = newv[upd]
        if nchg == 0: break
    return G_[ii, jj] == 1, nchg


def detect(F, shape, ii, jj, C0, prior, iw, il, blk, nmin, wfj, beta=2.0):
    """our water/land decision on one product grid (x2_two_knobs.py:run_product, LOOKS='est', class 5 kept from the PIXC):
    gamma likelihoods with block mean powers (blocks = 100 rare lines x 400 samples) started from PIXC classes 4/1 (3-4 / 1-2 if too few) and
    re-estimated twice from our own labels, effective looks from interior PIXC water / land, prior = PIXC prior_water_prob clipped to 0.05-0.95,
    Ising beta = 2, ICM from the MAP labels. Classes: water 4 (3 if a 4-neighbour is land), land 1 (2 if a 4-neighbour is water), 5 kept.
    Returns classes, water fraction (F - ml)/(mw - ml) clipped to 0-1 for the decided elements, and statistics."""
    _ub, BINV = np.unique(blk, return_inverse=True); nb = len(_ub)
    def bmeans(lw, ll):
        lw = lw & np.isfinite(F); ll = ll & np.isfinite(F)                          # one non-finite element must not void its block
        sw = np.bincount(BINV, weights=np.where(lw, F, 0.0), minlength=nb); nw = np.bincount(BINV, weights=lw.astype(float), minlength=nb)
        sl = np.bincount(BINV, weights=np.where(ll, F, 0.0), minlength=nb); nl = np.bincount(BINV, weights=ll.astype(float), minlength=nb)
        mw = np.where(nw >= nmin, sw / np.maximum(nw, 1), np.nan); ml = np.where(nl >= nmin, sl / np.maximum(nl, 1), np.nan); ok = mw > 1.5 * ml
        return np.where(ok, mw, np.nan)[BINV], np.where(ok, ml, np.nan)[BINV]
    F = np.asarray(F, float); mw1, ml1 = bmeans(C0 == 4, C0 == 1); mw2, ml2 = bmeans(np.isin(C0, (3, 4)), np.isin(C0, (1, 2)))
    MW0 = np.where(np.isfinite(mw1), mw1, mw2); ML0 = np.where(np.isfinite(ml1), ml1, ml2)
    a_ = iw & np.isfinite(MW0) & np.isfinite(F); b_ = il & np.isfinite(ML0) & np.isfinite(F)
    check(a_.sum() > 100 and b_.sum() > 100, f"too few interior water / land elements to estimate looks ({int(a_.sum())} / {int(b_.sum())})")
    LW = float(np.clip(1.0 / np.var(F[a_] / MW0[a_]), 1, 200)); LL = float(np.clip(1.0 / np.var(F[b_] / ML0[b_]), 1, 200))
    free = (C0 > 0) & (C0 != 5) & np.isfinite(F); isw0 = np.isin(C0, WATER); mw, ml = MW0, ML0; lab = isw0.copy(); map0 = lab.copy(); okf = free; last = 0
    for outer in range(2):
        Dv = nll(F, ml, LL) - nll(F, mw, LW) + np.log(prior / (1 - prior)); okf = free & np.isfinite(Dv)
        if outer == 0: lab = np.where(okf, Dv > 0, lab); map0 = lab.copy()
        lab, last = icm(shape, ii, jj, okf, lab, Dv, beta=beta)
        mwn, mln = bmeans(lab & (C0 != 5) & (C0 > 0), ~lab & (C0 > 0)); mw = np.where(np.isfinite(mwn), mwn, MW0); ml = np.where(np.isfinite(mln), mln, ML0)
    CROSS = ndimage.generate_binary_structure(2, 1); Gw = np.zeros(shape, bool); Gl = np.zeros(shape, bool); Gw[ii, jj] = lab; Gl[ii, jj] = ~lab
    nbw = ndimage.binary_dilation(Gw, CROSS)[ii, jj]; nbl = ndimage.binary_dilation(Gl, CROSS)[ii, jj]
    cc = C0.astype(np.uint8).copy(); w_ = okf & lab; l_ = okf & ~lab; cc[w_] = np.where(nbl[w_], 3, 4); cc[l_] = np.where(nbw[l_], 2, 1)
    wf = np.where(okf, np.clip((F - ml) / (mw - ml), 0, 1), wfj); wf = np.where(np.isfinite(wf), wf, wfj)
    st = dict(effective_looks_water=LW, effective_looks_land=LL, decided=int(okf.sum()), water_share_decided=float(lab[okf].mean()),
              agree_with_pixc_water_land=float((lab[okf] == isw0[okf]).mean()), map_without_mrf_agree=float((map0[okf] == isw0[okf]).mean()), icm_last_changes=int(last))
    check(set(np.unique(cc)) <= {0, 1, 2, 3, 4, 5, 6, 7}, "detection produced an unknown class")
    return cc, wf.astype(np.float32), st


# ---------------------------------------------------------------- drawing helpers (window figures)
def fit_cell_geometry(k, r, x, y):
    """quadratic model x, y = f(k, r) of the cell-centre positions in a window; returns a callable and the max fit residual (m)"""
    k0, r0 = np.mean(k), np.mean(r); dk, dr = (k - k0) / 100.0, (r - r0) / 100.0
    A = np.c_[np.ones_like(dk), dk, dr, dk * dk, dk * dr, dr * dr]; cx = np.linalg.lstsq(A, x, rcond=None)[0]; cy = np.linalg.lstsq(A, y, rcond=None)[0]
    def f(kq, rq):
        a, b = (np.asarray(kq, float) - k0) / 100.0, (np.asarray(rq, float) - r0) / 100.0; B = np.stack([np.ones_like(a), a, b, a * a, a * b, b * b], -1); return B @ cx, B @ cy
    fx, fy = f(k, r); return f, float(np.max(np.hypot(fx - x, fy - y)))


def cell_quads(f, k, r, half_k=0.5, half_r=0.5):
    """corners of cells (k, r) +- (half_k, half_r) from the fitted geometry: (n, 4, 2)"""
    out = np.zeros((len(k), 4, 2))
    for q, (a, b) in enumerate(((-half_k, -half_r), (-half_k, half_r), (half_k, half_r), (half_k, -half_r))):
        xx, yy = f(k + a, r + b); out[:, q, 0] = xx; out[:, q, 1] = yy
    return out


def cell_mesh(cx, cy, k, r, f):
    """watertight footprints of SLC cells (integer line k, range sample r) around their actual centres (cx, cy): every corner (k +- 1/2, r +- 1/2) is
    the mean, over the (up to 4) cells that share it, of (cell centre + offset to the corner along the fitted geometry f). Neighbours share corners,
    so nothing is left blank between them, and a cell moved by the water level moves its corners with it. Same corner order as cell_quads."""
    k = np.asarray(k, np.int64); r = np.asarray(r, np.int64); i = k - k.min(); j = r - r.min(); ni, nj = int(i.max()) + 2, int(j.max()) + 2
    check(len(np.unique(i * nj + j)) == len(k), "cell_mesh: duplicate (line, range) cells")
    kf, rf = k.astype(float), r.astype(float); xk1, yk1 = f(kf + 0.5, rf); xk0, yk0 = f(kf - 0.5, rf); xr1, yr1 = f(kf, rf + 0.5); xr0, yr0 = f(kf, rf - 0.5)
    jk = np.c_[xk1 - xk0, yk1 - yk0]; jr = np.c_[xr1 - xr0, yr1 - yr0]; c = np.c_[cx, cy]
    SX = np.zeros((ni, nj)); SY = np.zeros((ni, nj)); N = np.zeros((ni, nj)); offs = ((0, 0, -0.5, -0.5), (0, 1, -0.5, 0.5), (1, 1, 0.5, 0.5), (1, 0, 0.5, -0.5))
    for di, dj, a, b in offs:
        p = c + a * jk + b * jr; np.add.at(SX, (i + di, j + dj), p[:, 0]); np.add.at(SY, (i + di, j + dj), p[:, 1]); np.add.at(N, (i + di, j + dj), 1.0)
    MX = SX / np.maximum(N, 1.0); MY = SY / np.maximum(N, 1.0); out = np.zeros((len(k), 4, 2))
    for q, (di, dj, a, b) in enumerate(offs): out[:, q, 0] = MX[i + di, j + dj]; out[:, q, 1] = MY[i + di, j + dj]
    return out
def pseudo_pixels(x, y, area, along_unit, along_len):
    """PIXC footprints as used by the group ('pseudo pixels'): rectangle centred on the PIXC position, along_len along track, area/along_len across"""
    ua = np.asarray(along_unit, float) / np.linalg.norm(along_unit); uc = np.array([-ua[1], ua[0]]); wdt = np.asarray(area, float) / along_len
    c = np.c_[x, y]; ha = 0.5 * along_len * ua; out = np.zeros((len(x), 4, 2))
    for q, (sa, sc) in enumerate(((-1, -1), (-1, 1), (1, 1), (1, -1))): out[:, q] = c + sa * ha + sc * 0.5 * wdt[:, None] * uc
    return out


def rasterize(polys, values, n, origin=(0.0, 0.0), cell=1.0, order=None):
    """paint polygon values on an n x n grid (row 0 = south); later polygons in `order` overwrite earlier ones; NaN where no polygon"""
    from matplotlib.path import Path
    gc = (np.arange(n) + 0.5) * cell; GX, GY = np.meshgrid(origin[0] + gc, origin[1] + gc); P = np.c_[GX.ravel(), GY.ravel()]; V = np.full(n * n, np.nan)
    for i in (range(len(polys)) if order is None else order):
        q = polys[i]
        if not np.all(np.isfinite(q)): continue                       # e.g. a pixel without position or area
        c0 = max(int((q[:, 0].min() - origin[0]) / cell), 0); c1 = min(int((q[:, 0].max() - origin[0]) / cell) + 1, n)
        r0 = max(int((q[:, 1].min() - origin[1]) / cell), 0); r1 = min(int((q[:, 1].max() - origin[1]) / cell) + 1, n)
        if c1 <= c0 or r1 <= r0: continue
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1)); idx = (rr * n + cc).ravel(); idx = idx[Path(q).contains_points(P[idx])]; V[idx] = values[i]
    return V.reshape(n, n)


def disagreement(water, rs, drawn=None):
    """rs: 1 water, 0 land, -1 no label. missed = RiverScope water not drawn as water, as a share of RiverScope water; false = RiverScope land drawn
    as water, as a share of the drawn water. With `drawn` (footprint present), missed_no_pixel = the part of missed with no footprint at all.
    Shares are NaN when their base is empty (e.g. no cloud-free RiverScope water in the window)."""
    ok = rs >= 0; rw, rl = ok & (rs == 1), ok & (rs == 0); wat = np.asarray(water, bool)
    missed = int((rw & ~wat).sum()); false = int((rl & wat).sum()); nrw = int(rw.sum()); nwat = int((wat & ok).sum())
    out = dict(missed_m2=missed, false_m2=false, wrong_m2=missed + false, missed_pct=100.0 * missed / nrw if nrw else float("nan"),
               false_pct=100.0 * false / nwat if nwat else float("nan"), riverscope_water_m2=nrw, riverscope_labelled_m2=int(ok.sum()))
    if drawn is not None:
        gap = int((rw & ~np.asarray(drawn, bool)).sum()); out.update(missed_no_pixel_m2=gap, missed_no_pixel_pct=100.0 * gap / nrw if nrw else float("nan"))
    return out


def sample_labels(x, y, label, valid, transform):
    """label raster lookup (row 0 = north, GDAL affine (a, b, c, d, e, f)); returns 1 / 0 / -1"""
    a, b, c, d, e, f = transform; det = a * e - b * d
    col = ((x - c) * e - (y - f) * b) / det; row = (-(x - c) * d + (y - f) * a) / det
    ci = np.floor(col).astype(np.int64); ri = np.floor(row).astype(np.int64); ok = (ri >= 0) & (ri < label.shape[0]) & (ci >= 0) & (ci < label.shape[1])
    out = np.full(len(x), -1, np.int8); ii = np.where(ok)[0]; v = valid[ri[ii], ci[ii]]; out[ii[v]] = (label[ri[ii[v]], ci[ii[v]]] == 1).astype(np.int8); return out


def riverscope_sample(lat, lon, records):
    """RiverScope labels at lat/lon: records = [{'label': tif, 'sr': tif or None, 'udm2': tif or None}, ...]. A point takes its label from the first
    record that has a usable value there (so a later scene fills where an earlier one is cloudy or unlabelled). Usable = label 0/1 and, when given,
    Planet SR band 1 > 0 and UDM2 band 1 (clear) = 1 at that point, each raster read with its own transform (as x2_two_knobs.py). A record without
    SR or UDM2 is used without a cloud mask (the study always had both). Returns 1 water / 0 land / -1 unknown."""
    import rasterio, pyproj
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); out = np.full(len(lat), -1, np.int8)
    from rasterio.warp import transform_bounds
    for rec in records:
        todo = np.where((out < 0) & np.isfinite(lat) & np.isfinite(lon))[0]
        if len(todo) == 0: break
        with rasterio.open(rec["label"]) as d_:
            w_, s_, e_, n_ = transform_bounds(d_.crs, "EPSG:4326", *d_.bounds, densify_pts=21)
            if w_ <= e_:                                     # skip points outside the raster (not done across the antimeridian)
                todo = todo[(lon[todo] >= w_ - 0.01) & (lon[todo] <= e_ + 0.01) & (lat[todo] >= s_ - 0.01) & (lat[todo] <= n_ + 0.01)]
            if len(todo) == 0: continue
            lab = d_.read(1); T = d_.transform; crs = d_.crs
        tr = pyproj.Transformer.from_crs(4326, crs, always_xy=True); x, y = tr.transform(lon[todo], lat[todo]); x = np.asarray(x); y = np.asarray(y)
        v = sample_labels(x, y, lab, np.isin(lab, (0, 1)), (T.a, T.b, T.c, T.d, T.e, T.f))
        for key, test in (("sr", lambda a: a > 0), ("udm2", lambda a: a == 1)):
            if rec.get(key):
                with rasterio.open(rec[key]) as d2:
                    check(d2.crs == crs, f"{key} raster CRS differs from the label CRS: {rec[key]}")
                    a2 = d2.read(1); T2 = d2.transform
                ok2 = sample_labels(x, y, test(a2).astype(np.uint8), np.ones(a2.shape, bool), (T2.a, T2.b, T2.c, T2.d, T2.e, T2.f)) == 1
                v = np.where(ok2, v, -1)
        out[todo] = np.where(v >= 0, v, out[todo])
    return out
