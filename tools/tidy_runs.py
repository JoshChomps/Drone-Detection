"""
Sort C:/cuas_data/runs after a download from Drive. Moves only; never deletes.

1. Unpacks Drive zips found in runs/: a run folder's zip ("<run>/...") into runs/, a zip of loose files into runs/
   (a loose e12_eval_last.json goes into the folder of the one run the zip is about). An existing file is kept; a
   differing copy is reported, not overwritten. Each unpacked zip is moved to C:/cuas_data/_downloads/.
2. Moves each run's own loose reports into <run>/reports/: eval_, cmp_, e15_, e21_, e28_, e35_ files named after a run
   (e.g. e35_E24_s0_last.txt), and its training logs (<run>_train.log, <run>.log, <run>_resume1.log, ...).
   Only runs with a local folder; anything else (cross-run summaries such as e35_metrics_dut.txt, e36_*, e30_*,
   antiuav_test_brightness.json, which e35_metrics.py reads) stays at the top.

Scripts and the Colab queue still write to runs/ (top level); run this after copying their outputs in.

    python tools/tidy_runs.py            (dry run: prints what it would do)
    python tools/tidy_runs.py --apply
"""
import filecmp
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

D = Path(os.environ.get("CUAS_DATA", "C:/cuas_data"))
RUNS = D / "runs"
PARKED = D / "_downloads"
REPORT = re.compile(r"^(eval|cmp|e15|e21|e28|e35)_(?P<rest>.+)\.(txt|json)$")
LOG = re.compile(r"^(?P<rest>.+?)(_train|_resume\d+|_aborted\d+)?\.(log|err)$")


def run_of(stem, runs):
    """The longest run-folder name that `stem` starts with, followed by nothing or '_'."""
    hits = [r for r in runs if stem == r or stem.startswith(r + "_")]
    return max(hits, key=len) if hits else None


def unpack(z, apply, notes):
    with zipfile.ZipFile(z) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        top = {n.split("/")[0] for n in names}
        foldered = all("/" in n for n in names) and len(top) == 1
        about = None
        if not foldered:          # loose files: the run they are about, from names like e35_E24_s0_last.txt
            stems = {m.group("rest") for n in names for m in [REPORT.match(n) or LOG.match(n)] if m}
            runs = {s.split("_last")[0].removesuffix("_log") for s in stems}
            about = runs.pop() if len(runs) == 1 else None
        for n in names:
            dest = RUNS / n
            if not foldered and n == "e12_eval_last.json":
                if about is None:
                    notes.append(f"  ? {z.name}: loose {n}, run unknown; left in the zip")
                    continue
                dest = RUNS / about / n
            if dest.exists():
                if apply:
                    tmp = PARKED / "_cmp.tmp"
                    tmp.write_bytes(zf.read(n))
                    same = filecmp.cmp(tmp, dest, shallow=False)
                    tmp.unlink()
                    if not same:
                        notes.append(f"  ! {z.name}: {n} differs from the existing {dest.relative_to(RUNS)}; kept the existing")
                continue
            print(f"  unpack {z.name} : {n} -> {dest.relative_to(RUNS)}")
            if apply:
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(n))
    print(f"  park   {z.name} -> {PARKED}")
    if apply:
        shutil.move(str(z), str(PARKED / z.name))


def main(apply):
    if apply:
        PARKED.mkdir(exist_ok=True)
    notes = []
    for z in sorted(RUNS.glob("*.zip")):
        unpack(z, apply, notes)
    runs = sorted(p.name for p in RUNS.iterdir() if p.is_dir() and (p / "weights").is_dir()
                  or p.is_dir() and any(p.glob("dets_*.json")) or p.is_dir() and any(p.glob("e12_eval*.json")))
    moved = []
    for f in sorted(p for p in RUNS.iterdir() if p.is_file()):
        m = REPORT.match(f.name) or LOG.match(f.name)
        run = run_of(m.group("rest"), runs) if m else None
        if run is None:
            continue
        dest = RUNS / run / "reports" / f.name
        if dest.exists():
            notes.append(f"  ! {f.name}: {dest.relative_to(RUNS)} already exists; left at the top")
            continue
        moved.append((f.name, f"{run}/reports/{f.name}"))
        print(f"  move   {f.name} -> {run}/reports/")
        if apply:
            dest.parent.mkdir(exist_ok=True)
            shutil.move(str(f), str(dest))
    left = sorted(p.name for p in RUNS.iterdir() if p.is_file() and p.name not in {a for a, _ in moved})
    print(f"\n{len(moved)} reports {'moved' if apply else 'to move'}; {len(left)} files stay at the top:")
    print("  " + "\n  ".join(left))
    print("\n".join(notes) or "no conflicts")
    if not apply:
        print("\nDry run. Re-run with --apply.")


if __name__ == "__main__":
    main("--apply" in sys.argv)
