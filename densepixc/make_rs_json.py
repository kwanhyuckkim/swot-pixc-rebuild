#!/usr/bin/env python3
"""Write the RiverScope list that dpc_run_tile.py / dpc_plot.py read, from the slc_to_pixc study's aux/rs_files.json.
Output: [{"label": RiverScope label tif, "sr": Planet SR tif, "udm2": Planet UDM2 tif, "reach": ..., "hours_from_swot": ...}, ...]
sorted by |hours from the SWOT pass| (the closest scene first). Records without both Planet SR and UDM2 (no cloud mask) or with a missing file
are skipped, as in the study (x2_two_knobs.py). Exit code 3 (and no file) when the tile has no usable record.
Usage: python make_rs_json.py --rs-files aux/rs_files.json --tile 037L --out rs_037L.json"""
import argparse, json, os, sys

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--rs-files", required=True); ap.add_argument("--tile", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
with open(a.rs_files) as f: recs = json.load(f)
out = []
for r in sorted(recs.get(a.tile, []), key=lambda r: abs(float(r["hours_from_swot"])) if r.get("hours_from_swot") is not None else float("inf")):   # unknown time last
    files = {k: r.get(k) for k in ("file", "planet_sr", "planet_udm2")}
    missing = [k for k, v in files.items() if not v or not os.path.exists(v)]
    if missing:
        print(f"skip reach {r.get('reach')}: no file for {missing}"); continue
    out.append(dict(label=files["file"], sr=files["planet_sr"], udm2=files["planet_udm2"], reach=r.get("reach"), hours_from_swot=r.get("hours_from_swot")))
if not out:
    print(f"no usable RiverScope record for tile {a.tile} in {a.rs_files}"); sys.exit(3)
with open(a.out, "w") as f: json.dump(out, f, indent=1)
print(f"wrote {a.out}: {len(out)} records", [(r["reach"], r["hours_from_swot"]) for r in out])
