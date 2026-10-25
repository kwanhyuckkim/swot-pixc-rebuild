#!/usr/bin/env python3
"""Unit tests for dpc_core with synthetic data (numpy, scipy, matplotlib only). Run: python tests/test_dpc_core.py  (or pytest)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dpc_core as C


def test_geodesy_round_trip():
    lat = np.array([65.3, -12.0, 0.0]); lon = np.array([-157.5, 30.0, 179.9]); h = np.array([50.0, -20.0, 3000.0])
    la, lo, hh = C.ecef2geod(C.geod2ecef(lat, lon, h))
    assert np.allclose(la, lat, rtol=0, atol=1e-9) and np.allclose(lo, lon, rtol=0, atol=1e-9) and np.allclose(hh, h, rtol=0, atol=1e-4)


def test_dense_cells_mapping():
    ai = np.array([0, 0, 5, 9]); ri = np.array([2, 3, 2, 7]); L, OFF, NL = 7, 4, 66
    K, RR, PID, D = C.dense_cells(ai, ri, L, OFF, NL)
    assert np.all(np.diff(K) >= 0)
    assert np.all((K - OFF - D) == ai[PID] * L)                       # every cell maps back to its rare line
    assert np.all(RR == ri[PID]) and np.all((D >= -3) & (D <= 3))
    # rare line 9: centre 67 -> lines 64..70, only 64, 65 exist (NL = 66)
    assert set(K[PID == 3]) == {64, 65}
    assert (PID == 0).sum() == 7 and set(K[PID == 0]) == {1, 2, 3, 4, 5, 6, 7}


def test_rare_and_sliding_multilook_agree_on_centre_lines():
    rng = np.random.default_rng(1); NA, NP, L, OFF = 12, 6, 7, 4; NL = NA * L + 8
    ai, ri = np.meshgrid(np.arange(NA), np.arange(NP), indexing="ij"); ai = ai.ravel(); ri = ri.ravel()
    K, RR, PID, D = C.dense_cells(ai, ri, L, OFF, NL); v = rng.gamma(2.0, 1.0, len(K))
    rare, cnt = C.rare_mean(v, PID, len(ai)); sl, n = C.sliding_mean(v, K, RR, NL, NP, L)
    m = (D == 0) & (cnt[PID] == L) & (n == L); assert m.sum() > 0
    assert np.allclose(sl[m], rare[PID[m]], rtol=1e-6)
    # block-filled = rare value on every line of the pixel
    blk = rare[PID]; assert np.all(blk[PID == 5] == rare[5])
    # sliding window at an interior line = mean of lines k-3..k+3 at the same range sample
    k = int(K[(D == 2) & (PID == 20)][0]); r = int(RR[(D == 2) & (PID == 20)][0]); w = (RR == r) & (np.abs(K - k) <= 3)
    assert np.isclose(sl[(K == k) & (RR == r)][0], v[w].mean(), rtol=1e-6)


def test_multilooks_leave_out_non_finite_values():
    K = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6]); RR = np.array([0] * 7 + [1] * 7); PID = np.array([0] * 7 + [1] * 7)
    v = np.arange(14, dtype=float); v[3] = np.nan
    m, c = C.rare_mean(v, PID, 2); assert np.isclose(m[0], np.mean([0, 1, 2, 4, 5, 6])) and c[0] == 6 and np.isclose(m[1], 10.0) and c[1] == 7
    o = np.lexsort((RR, K)); sl, n = C.sliding_mean(v[o], K[o], RR[o], 7, 2, 7); sl = sl[np.argsort(o)]; n = n[np.argsort(o)]
    assert np.all(np.isfinite(sl[np.arange(14) != 3])) and n[6] == 3 and np.isclose(sl[6], np.mean([4, 5, 6]))       # the NaN does not spread down the column


def test_read_lines_any_chunk():
    class _T: pass
    t = _T(); rng = np.random.default_rng(6); NL, NP = 40, 9; zp = rng.normal(size=(NL, NP, 2)).astype(np.float32); zm = rng.normal(size=(NL, NP, 2)).astype(np.float32)
    t._slc = {"slc": {"slc_plus_y": zp, "slc_minus_y": zm}}; t.NL = NL
    K, RR, PID, D = C.dense_cells(np.array([0, 2, 4, 5]), np.array([1, 8, 0, 3]), 7, 4, NL)
    for ch in (1, 3, 7, 1500):
        Pp, Pm, J = C.read_lines(t, K, RR, chunk=ch); a = zp[K, RR, 0] + 1j * zp[K, RR, 1]; b = zm[K, RR, 0] + 1j * zm[K, RR, 1]
        assert np.allclose(Pp, np.abs(a) ** 2, rtol=1e-6) and np.allclose(Pm, np.abs(b) ** 2, rtol=1e-6) and np.allclose(J, a * np.conj(b), rtol=1e-5, atol=1e-6)


def test_disagreement_no_pixel_and_no_label():
    rs = np.full((10, 10), -1, np.int8); s = C.disagreement(np.zeros((10, 10), bool), rs)
    assert np.isnan(s["missed_pct"]) and np.isnan(s["false_pct"]) and s["riverscope_labelled_m2"] == 0      # nothing labelled: no score, not 0 %
    rs = np.zeros((10, 10), np.int8); rs[:, :5] = 1; water = np.zeros((10, 10), bool); water[:, :2] = True; drawn = np.zeros((10, 10), bool); drawn[:, :4] = True
    s = C.disagreement(water, rs, drawn=drawn); assert s["missed_m2"] == 30 and s["missed_no_pixel_m2"] == 10 and np.isclose(s["missed_pct"], 60.0) and s["false_m2"] == 0


def test_rasterize_skips_polygons_without_position():
    sq = np.array([[[1, 1], [1, 3], [3, 3], [3, 1]], [[np.nan, 0], [0, 1], [1, 1], [1, 0]]], float); v = C.rasterize(sq, np.array([1.0, 2.0]), 5)
    assert np.nansum(v) == 4


def test_sliding_mean_without_finite_lines_is_nan():
    K = np.arange(10); RR = np.zeros(10, int); v = np.r_[np.full(8, np.nan), 1.0, 2.0]
    sl, n = C.sliding_mean(v, K, RR, 10, 1, 7); assert np.all(np.isnan(sl[:5])) and np.all(n[:5] == 0) and np.isclose(sl[9], 1.5)   # 0 lines -> NaN, not 0


def test_sliding_mean_min_lines():
    K = np.arange(14); RR = np.zeros(14, int); v = np.r_[np.full(7, 1.0), np.full(7, np.nan)]          # one pixel with values, then 7 empty lines
    sl, n = C.sliding_mean(v, K, RR, 14, 1, 7, min_lines=4)
    assert list(n[:10]) == [4, 5, 6, 7, 6, 5, 4, 3, 2, 1] and np.all(np.isfinite(sl[:7])) and np.all(np.isnan(sl[7:]))   # every line of the pixel has >= 4 lines


def test_landstop_mean_slides_away_from_land():
    NL = 20; K = np.arange(NL); RR = np.zeros(NL, int); v = np.arange(NL, dtype=float)
    bar = np.zeros((NL, 1), bool); bar[10, 0] = True                                     # one land line at 10
    m, n, isb = C.landstop_mean(v, K, RR, NL, 1, 7, bar, min_lines=4)
    assert isb[10] and m[10] == 10.0 and n[10] == 1                                       # a land line keeps its own value
    assert np.isclose(m[9], np.mean(range(3, 10))) and n[9] == 7                           # window slid to 3..9, not 6..12
    assert np.isclose(m[11], np.mean(range(11, 18))) and np.isclose(m[5], np.mean(range(2, 9)))
    bar2 = np.zeros((NL, 1), bool); bar2[[5, 8], 0] = True                                 # a run of two lines (6, 7) between land lines
    m2, n2, _ = C.landstop_mean(v, K, RR, NL, 1, 7, bar2, min_lines=4)
    assert np.isnan(m2[6]) and np.isnan(m2[7]) and n2[6] == 2                              # fewer than 4 lines: not filled
    m3, _, _ = C.landstop_mean(v, K, RR, NL, 1, 7, np.zeros((NL, 1), bool), min_lines=4)
    s3, _ = C.sliding_mean(v, K, RR, NL, 1, 7, min_lines=4); ok = np.isfinite(s3) & (K >= 3) & (K <= 16)
    assert np.allclose(m3[ok], s3[ok])                                                     # no land: same as the plain moving average inside


def test_isolated_water_to_land():
    G = np.ones((12, 12), np.uint8); G[2:4, 2:4] = 3; G[6:9, 6:9] = 3; G[7, 7] = 4; G[2, 4] = 2; G[0:2, 10] = 3   # 3-only patch, patch with a 4, edge patch
    out = C.isolated_water_to_land(G)
    assert np.all(out[2:4, 2:4] == 1) and out[2, 4] == 1                                   # isolated class-3 patch and its class-2 cell -> land
    assert np.all(out[6:9, 6:9] == G[6:9, 6:9]) and np.all(out[0:2, 10] == 3)              # a patch with class 4, and one at the edge, stay


def test_detect_one_nan_does_not_void_a_block_and_prior_counts():
    rng = np.random.default_rng(7); NA, NP = 100, 60; ai, ri = np.meshgrid(np.arange(NA), np.arange(NP), indexing="ij"); ai = ai.ravel(); ri = ri.ravel()
    water = (ri >= 20) & (ri < 40); F = np.where(water, rng.gamma(4.0, 25.0, ai.size), rng.gamma(4.0, 1.0, ai.size)); C0 = np.where(water, 4, 1).astype(np.uint8)
    a = ((NA, NP), ai, ri, C0, np.full(ai.size, 0.5), water & (ri > 22) & (ri < 37), ~water & ((ri < 17) | (ri > 42)), (ai // 100) * 100000 + ri // 400, 20, np.zeros(ai.size))
    _, _, s0 = C.detect(F, *a); F1 = F.copy(); F1[5] = np.nan; _, _, s1 = C.detect(F1, *a)
    assert s1["decided"] == s0["decided"] - 1                                         # only the NaN element is left undecided
    C1 = np.where(ri < 30, 4, 1).astype(np.uint8); pri = np.where(ri < 30, 0.95, 0.05)                          # PIXC water left, land right
    Fa = np.where(ri < 30, rng.gamma(4.0, 10.0, ai.size), rng.gamma(4.0, 1.0, ai.size))
    band = (ri >= 20) & (ri < 40); Fa[band] = rng.gamma(4.0, 2.5, int(band.sum()))                                # the same in-between values on both sides
    a2 = ((NA, NP), ai, ri, C1, pri, (ri > 2) & (ri < 17), (ri > 43) & (ri < 57), (ai // 100) * 100000 + ri // 400, 20, np.zeros(ai.size))
    c, _, _ = C.detect(Fa, *a2, beta=0.0); wat = np.isin(c, C.WATER)
    assert wat[(ri >= 20) & (ri < 30)].mean() > wat[(ri >= 30) & (ri < 40)].mean() + 0.2  # same kind of data: more water where the prior says water


def test_complex_rare_mean():
    PID = np.array([0, 0, 1]); v = np.array([1 + 1j, 3 - 1j, 2j]); m, c = C.rare_mean(v, PID, 2)
    assert np.allclose(m, [2 + 0j, 2j]) and np.all(c == [2, 1])


def test_icm_cleans_noise():
    rng = np.random.default_rng(2); shape = (60, 60); truth = np.zeros(shape, bool); truth[:, 25:40] = True
    Dv = np.where(truth, 1.0, -1.0) + rng.normal(0, 1.5, shape); ii, jj = np.meshgrid(np.arange(60), np.arange(60), indexing="ij"); ii = ii.ravel(); jj = jj.ravel()
    free = np.ones(ii.size, bool); lab0 = Dv.ravel() > 0; lab, _ = C.icm(shape, ii, jj, free, lab0, Dv.ravel(), beta=2.0)
    assert (lab == truth.ravel()).mean() > (lab0 == truth.ravel()).mean() + 0.05


def test_nll_prefers_right_mean():
    F = np.array([10.0]); assert C.nll(F, 10.0, 4.0)[0] < C.nll(F, 1.0, 4.0)[0]


def test_detect_recovers_two_populations():
    rng = np.random.default_rng(3); NA, NP = 120, 60; ai, ri = np.meshgrid(np.arange(NA), np.arange(NP), indexing="ij"); ai = ai.ravel(); ri = ri.ravel()
    water = (ri >= 20) & (ri < 40); F = np.where(water, rng.gamma(4.0, 25.0, ai.size), rng.gamma(4.0, 1.0, ai.size))
    C0 = np.where(water, 4, 1).astype(np.uint8); C0[rng.random(ai.size) < 0.05] = 3                         # noisy starting classes
    cls, wf, st = C.detect(F, (NA, NP), ai, ri, C0, np.full(ai.size, 0.5), water & (ri > 22) & (ri < 37), ~water & ((ri < 17) | (ri > 42)),
                           (ai // 100) * 100000 + ri // 400, 20, np.zeros(ai.size))
    assert (np.isin(cls, C.WATER) == water).mean() > 0.99 and st["effective_looks_water"] > 2
    assert set(np.unique(cls)) <= {1, 2, 3, 4} and (cls == 3).any() and (cls == 2).any()                 # edge classes next to the other kind
    edge_w = water & ((ri == 20) | (ri == 39)); assert (cls[edge_w] == 3).mean() > 0.9


def test_detect_keeps_class5_and_the_mrf_helps():
    rng = np.random.default_rng(5); NA, NP = 160, 80; ai, ri = np.meshgrid(np.arange(NA), np.arange(NP), indexing="ij"); ai = ai.ravel(); ri = ri.ravel()
    water = (ri >= 25) & (ri < 55); F = np.where(water, rng.gamma(1.0, 4.0, ai.size), rng.gamma(1.0, 1.0, ai.size))        # single look: noisy
    C0 = np.where(water, 4, 1).astype(np.uint8); dark = water & (ai < 10); C0[dark] = 5
    args = ((NA, NP), ai, ri, C0, np.full(ai.size, 0.5), water & (ri > 30) & (ri < 50), ~water & ((ri < 20) | (ri > 60)), (ai // 100) * 100000 + ri // 400, 20, np.zeros(ai.size))
    c2, _, _ = C.detect(F, *args, beta=2.0); c0, _, _ = C.detect(F, *args, beta=0.0)
    assert np.all(c2[dark] == 5) and np.all(c0[dark] == 5)                                                       # dark water is not re-decided
    ok = ~dark; a2 = (np.isin(c2, C.WATER) == water)[ok].mean(); a0 = (np.isin(c0, C.WATER) == water)[ok].mean()
    assert a2 > a0 + 0.05, (a2, a0)


def test_level_grid_median():
    rng = np.random.default_rng(4); x = rng.uniform(0, 2000, 8000); y = rng.uniform(0, 2000, 8000); h = 50 + 0.005 * x + rng.normal(0, 0.05, 8000)
    gr = C.level_grid(x, y, h, np.ones(8000, bool), rmax=300, k=64, cell=100, pad=200); v = C.grid_bilinear(gr, np.array([1500.0, 500.0]), np.array([500.0, 1500.0]))
    assert abs(v[0] - 57.5) < 0.1 and abs(v[1] - 52.5) < 0.1            # an x/y swap gives 52.5 / 57.5; half a grid cell of shift gives 0.25 m


def test_pseudo_pixel_area_and_orientation():
    q = C.pseudo_pixels(np.array([0.0]), np.array([0.0]), np.array([440.0]), [0.0, 1.0], 22.0)[0]
    area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], 1)) - np.dot(q[:, 1], np.roll(q[:, 0], 1))); assert np.isclose(area, 440.0)
    assert np.isclose(q[:, 1].max() - q[:, 1].min(), 22.0) and np.isclose(q[:, 0].max() - q[:, 0].min(), 20.0)


def test_cell_quads_tile_the_plane():
    k, r = np.meshgrid(np.arange(30, dtype=float), np.arange(20, dtype=float), indexing="ij"); k = k.ravel(); r = r.ravel()
    x = 3.0 * k + 0.5 * r; y = -1.0 * k + 15.0 * r; f, res = C.fit_cell_geometry(k, r, x, y); assert res < 1e-6
    q = C.cell_quads(f, k, r); i = int(np.where((k == 5) & (r == 5))[0][0]); j = int(np.where((k == 6) & (r == 5))[0][0])
    assert np.allclose(q[i, 2], q[j, 1]) and np.allclose(q[i, 3], q[j, 0])       # shared edge between line k and k+1


def test_cell_mesh_is_watertight():
    k, r = np.meshgrid(np.arange(10), np.arange(8), indexing="ij"); k = k.ravel(); r = r.ravel()
    x = 3.0 * k + 0.5 * r; y = -1.0 * k + 15.0 * r; f, _ = C.fit_cell_geometry(k.astype(float), r.astype(float), x, y)
    assert np.allclose(C.cell_mesh(x, y, k, r, f), C.cell_quads(f, k.astype(float), r.astype(float)))       # exact geometry: same as the fitted mesh
    x2 = x.copy(); x2[(k == 4) & (r == 3)] += 2.0; q = C.cell_mesh(x2, y, k, r, f)                         # one cell moved by 2 m
    i = int(np.where((k == 4) & (r == 3))[0][0]); j = int(np.where((k == 5) & (r == 3))[0][0])
    assert np.allclose(q[i, 2], q[j, 1]) and np.allclose(q[i, 3], q[j, 0])                                # neighbours still share corners
    assert np.isclose(q[i, :, 0].mean() - C.cell_quads(f, k.astype(float), r.astype(float))[i, :, 0].mean(), 0.5)   # each corner is shared by 4 cells: moves 2/4 m
    try:
        C.cell_mesh(np.r_[x, x[:1]], np.r_[y, y[:1]], np.r_[k, k[:1]], np.r_[r, r[:1]], f)
    except C.CheckError:
        pass
    else:
        raise AssertionError("duplicate cells not caught")


class _Orbit:
    """straight polar orbit at 904 km, left-looking (side +1), SLC near range 903,418 m and 0.75 m samples, as tile 037L"""
    def __init__(self):
        n = 2000; lat = np.linspace(64.0, 66.0, n); self.S = C.geod2ecef(lat, np.full(n, -157.4), np.full(n, 904300.0)); self.V = np.gradient(self.S, axis=0) * 2000.0
        self.SF, self.side, self.NR_SLC, self.DR = 0, 1.0, 903418.6, 0.75


def test_geoloc_uses_record_k_plus_sf():
    t = _Orbit(); k = np.array([700.0]); r = np.array([2000.0]); h = np.array([30.0])
    la0, lo0, X0 = C.geoloc(t, k, r, h, np.full(1, 2.3)); t.SF = 100; la1, lo1, X1 = C.geoloc(t, k - 100.0, r, h, np.full(1, 2.3))
    assert np.allclose(X0, X1, atol=1e-6)                                      # line k - SF with SF = 100 is the same TVP record as line k with SF = 0


def test_geoloc_hits_the_height_and_flags_no_root():
    t = _Orbit(); k = np.array([500.0, 900.0, 1200.0, 1500.0]); r = np.array([1500.0, 3000.0, 4600.0, 600.0]); h = np.array([50.0, 120.0, 0.0, 400.0])
    la, lo, X = C.geoloc(t, k, r, h, np.full(4, 2.3))
    ok = np.isfinite(la); assert ok[:3].all() and not ok[3]                     # sample 600 at 400 m: the range sphere stays above 400 m even at nadir -> NaN
    assert np.allclose(C.ecef2geod(X[ok])[2], h[ok], rtol=0, atol=1e-3)           # the point lies at the requested height
    R = t.NR_SLC + t.DR * r[ok] + 2.3 + C.RANGE_BIAS_M; Sj, dn, cl = C.frame(t, k[ok])
    assert np.allclose(np.linalg.norm(X[ok] - Sj, axis=1), R, rtol=0, atol=1e-3)  # ... at the requested range (media and 1.23 cm included)
    assert np.all(lo[ok] < -157.4)                                                # ... west of the northbound track: the left swath


def test_rasterize_and_disagreement():
    sq = np.array([[[10, 10], [10, 30], [30, 30], [30, 10]]], float); v = C.rasterize(sq, np.array([1.0]), 40)
    assert np.nansum(v) == 400
    rs = np.zeros((40, 40), np.int8); rs[10:30, 10:20] = 1; rs[0, 0] = -1; s = C.disagreement(np.nan_to_num(v) > 0.5, rs)
    assert s["missed_m2"] == 0 and s["false_m2"] == 200 and s["wrong_m2"] == 200


def test_sample_labels_north_up_affine():
    lab = np.array([[1, 0], [1, 1]], np.uint8); valid = np.ones((2, 2), bool); T = (3.0, 0.0, 100.0, 0.0, -3.0, 206.0)    # x0 100, y0 206 (north)
    v = C.sample_labels(np.array([101.0, 104.0, 101.0, 110.0]), np.array([205.0, 205.0, 202.0, 205.0]), lab, valid, T)
    assert list(v) == [1, 0, 1, -1]                                               # asymmetric: a row/column swap would give 1, 1, 0


def test_check_raises():
    try:
        C.check(False, "boom")
    except C.CheckError as e:
        assert "boom" in str(e)
    else:
        raise AssertionError("check did not raise")


import unittest


class Skip(unittest.SkipTest):                                     # pytest and unittest show it as skipped
    pass


def _need(mod):
    try:
        return __import__(mod)
    except ImportError:
        raise Skip(f"{mod} not installed")


def test_riverscope_sample_masks_and_order():
    rasterio = _need("rasterio"); import tempfile
    from rasterio.transform import from_origin
    d = tempfile.mkdtemp(); T = from_origin(500000.0, 7000030.0, 3.0, 3.0)                    # UTM 32604, 10 x 10 pixels of 3 m
    def tif(name, arr):
        p = os.path.join(d, name)
        with rasterio.open(p, "w", driver="GTiff", height=10, width=10, count=1, dtype="uint8", crs="EPSG:32604", transform=T) as f: f.write(arr.astype(np.uint8), 1)
        return p
    lab = np.zeros((10, 10)); lab[:, 5:] = 1; lab[0, 0] = 255; sr = np.ones((10, 10)); sr[9, 9] = 0; udm = np.ones((10, 10)); udm[9, 0] = 0
    r1 = dict(label=tif("l1.tif", lab), sr=tif("s1.tif", sr), udm2=tif("u1.tif", udm)); r2 = dict(label=tif("l2.tif", np.ones((10, 10))), sr=None, udm2=None)
    import pyproj; tr = pyproj.Transformer.from_crs(32604, 4326, always_xy=True)
    xs = 500000.0 + 3.0 * np.array([0.5, 7.5, 9.5, 0.5, 2.5]); ys = 7000030.0 - 3.0 * np.array([0.5, 4.5, 9.5, 9.5, 4.5]); lo, la = tr.transform(xs, ys)
    v1 = C.riverscope_sample(np.asarray(la), np.asarray(lo), [r1]); assert list(v1) == [-1, 1, -1, -1, 0]      # no label, water, SR = 0, UDM2 = 0, land
    v2 = C.riverscope_sample(np.asarray(la), np.asarray(lo), [r1, r2]); assert list(v2) == [1, 1, 1, 1, 0]    # the second record fills where the first is unusable


def test_phi_ref_lines_nan_outside_grdem():
    _need("pyproj"); t = _Orbit(); n = len(t.S); t.T = np.arange(n, dtype=float); t.GT = t.T.copy(); t.LAM = 0.008385803
    la, lo, _ = C.ecef2geod(t.S); t.GLA, t.GLO = la, lo; t.GV = t.V.copy(); t.SCT = 10000.0 + 50.0 * np.arange(1000); t.GH = np.zeros((n, 1000))   # 10-60 km, flat
    off = np.array([0.0, 0.0, 6.0]); t.Ap = t.S + off; t.Am = t.S - off
    k = np.array([600, 600, 600]); r = np.array([-80000.0, 3000.0, 200000.0])                       # near nadir / inside / beyond 60 km
    ph = C.phi_ref_lines(t, k, r); assert np.isnan(ph[0]) and np.isfinite(ph[1]) and np.isnan(ph[2])


if __name__ == "__main__":
    n = 0; skipped = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); n += 1; print("ok", name)
            except Skip as e:
                skipped.append(name); print("skip", name, "-", e)
    print(f"{n} tests passed" + (f", {len(skipped)} skipped (missing packages)" if skipped else ""))
