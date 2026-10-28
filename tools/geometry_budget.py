#!/usr/bin/env python3
"""
geometry_budget.py
Camera budget for a site, from the formulas in 13_system_requirements_and_geometry_budget.md (sections 2-3).

For each threat: the range needed so an alert completes before the drone reaches the zone edge,
the widest lens that still puts min_px on it there, and where it appears in elevation. Then one
long-range camera row (the narrowest lens any in-scope threat needs), tilted with its lower edge at
the skyline, and what that row delivers per threat.

    python tools/geometry_budget.py --scenario A
    python tools/geometry_budget.py --scenario B --timeline_s 5
    python tools/geometry_budget.py --site my_site.json   # same shape as an entry in SCENARIOS
"""
import argparse
import json
import math

# Threat sizes, speeds and heights are working assumptions (README threat table); replace per study.
THREATS = [
    {"name": "Mini-class quad", "size_m": 0.25, "speed_mps": 15, "height_m": 30},
    {"name": "Mavic-class quad", "size_m": 0.35, "speed_mps": 15, "height_m": 100},
    {"name": "FPV", "size_m": 0.30, "speed_mps": 40, "height_m": 30},
    {"name": "FPV, low", "size_m": 0.30, "speed_mps": 40, "height_m": 5},
    {"name": "Shahed-class", "size_m": 2.5, "speed_mps": 50, "height_m": 100},
]
SITE_DEFAULTS = {"skyline_deg": 2.0, "mast_m": 5.0, "min_px": 10, "width_px": 3840, "aspect": 16 / 9}
SCENARIOS = {
    "A": {"name": "Test field", "zone_radius_m": 50, "sector_deg": 90, "timeline_s": 5, "threats": THREATS},
    "B": {"name": "Perimeter", "zone_radius_m": 150, "sector_deg": 180, "timeline_s": 10, "threats": THREATS},
    "C": {"name": "360 deg rooftop", "zone_radius_m": 100, "sector_deg": 360, "timeline_s": 5, "threats": THREATS},
}
OVERRIDES = ("zone_radius_m", "sector_deg", "timeline_s", "skyline_deg", "mast_m", "min_px")


def f_px(hfov_deg, width_px=3840):
    return (width_px / 2) / math.tan(math.radians(hfov_deg / 2))


def hfov_deg(f, width_px=3840):
    return 2 * math.degrees(math.atan((width_px / 2) / f))


def vfov_deg(hfov, aspect=16 / 9):
    return 2 * math.degrees(math.atan(math.tan(math.radians(hfov / 2)) / aspect))


def budget(site: dict) -> dict:
    s = {**SITE_DEFAULTS, **site}
    rows = []
    for t in s["threats"]:
        d = t["speed_mps"] * s["timeline_s"] + s["zone_radius_m"]      # horizontal distance at first detection
        dh = t["height_m"] - s["mast_m"]
        slant = math.hypot(d, dh)
        el = math.degrees(math.atan2(dh, d))
        f_need = slant * s["min_px"] / t["size_m"]
        rows.append({**t, "need_m": d, "slant_m": slant, "elev_deg": el, "below_skyline": el < s["skyline_deg"],
                     "max_hfov_deg": hfov_deg(f_need, s["width_px"])})

    in_scope = [r for r in rows if not r["below_skyline"]]
    hfov = min(r["max_hfov_deg"] for r in in_scope) if in_scope else None
    if hfov:
        f, vfov = f_px(hfov, s["width_px"]), vfov_deg(hfov, s["aspect"])
        for r in rows:
            dh = r["height_m"] - s["mast_m"]
            if dh <= 0:
                continue    # never rises above the camera, let alone the skyline
            # Approaching, the row sees the target at horizontal distances where it has >= min_px (d <= reach)
            # and sits inside the row's elevations (d_top <= d <= d_low). The first detection is the largest such d.
            reach = math.sqrt(max((f * r["size_m"] / s["min_px"]) ** 2 - dh ** 2, 0))
            d_low = dh / math.tan(math.radians(s["skyline_deg"])) if s["skyline_deg"] > 0 else math.inf
            d_top = dh / math.tan(math.radians(s["skyline_deg"] + vfov))
            first = min(reach, d_low)
            if first >= d_top:
                r["warning_s"] = max(first - s["zone_radius_m"], 0) / r["speed_mps"]
                r["in_row_s"] = (first - d_top) / r["speed_mps"]    # time before it climbs out of the row
    return {"site": s, "threats": rows, "row_hfov_deg": hfov,
            "row_cameras": math.ceil(s["sector_deg"] / hfov) if hfov else None,
            "row_elev_deg": (s["skyline_deg"], s["skyline_deg"] + vfov_deg(hfov, s["aspect"])) if hfov else None}


def report(b: dict) -> str:
    s = b["site"]
    out = [f"{s.get('name', 'Site')}: zone {s['zone_radius_m']} m, sector {s['sector_deg']} deg, timeline "
           f"{s['timeline_s']} s, skyline {s['skyline_deg']} deg, mast {s['mast_m']} m, >= {s['min_px']} px",
           f"{'threat':18} {'need m':>7} {'slant m':>7} {'elev':>6} {'max HFOV':>9} {'warning s':>9} {'in row s':>8}  note"]
    for r in b["threats"]:
        if r["below_skyline"]:
            note = ("below skyline at the needed range: seen late" if r.get("warning_s")
                    else "below skyline: out of optical scope")
        elif "warning_s" in r or not b["row_hfov_deg"]:
            note = ""
        else:
            note = f"row never sees it: add an upper row <= {r['max_hfov_deg']:.0f} deg"
        warn = (f"{r['warning_s']:9.1f} {r['in_row_s']:8.1f}" if "warning_s" in r else f"{'-':>9} {'-':>8}")
        out.append(f"{r['name']:18} {r['need_m']:7.0f} {r['slant_m']:7.0f} {r['elev_deg']:5.1f}d "
                   f"{r['max_hfov_deg']:8.1f}d {warn}  {note}")
    if b["row_hfov_deg"]:
        lo, hi = b["row_elev_deg"]
        out.append(f"Long-range row: {b['row_cameras']} cameras ({s['width_px']} px wide) at <= {b['row_hfov_deg']:.1f} deg HFOV, "
                   f"covering elevations {lo:.1f}-{hi:.1f} deg")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Camera budget for a C-UAS site (doc 13).")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="A")
    parser.add_argument("--site", help="JSON file with the same keys as a SCENARIOS entry (overrides --scenario)")
    for key in OVERRIDES:
        parser.add_argument(f"--{key}", type=float, help=f"override {key}")
    args = parser.parse_args()

    site = dict(SCENARIOS[args.scenario])
    if args.site:
        with open(args.site, encoding="utf-8") as f:
            site = {"threats": THREATS, **json.load(f)}
    site.update({k: getattr(args, k) for k in OVERRIDES if getattr(args, k) is not None})
    print(report(budget(site)))


if __name__ == "__main__":
    main()
