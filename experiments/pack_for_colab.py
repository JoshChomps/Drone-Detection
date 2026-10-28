"""
Bundle what a Colab run of E14 needs (training data, evaluation sets, code) into two zips for Google Drive.

    python experiments/pack_for_colab.py        (~1.4 GB of data; a few minutes; CPU only)
    python experiments/pack_for_colab.py --code (code zip only: after code changes, when Drive already has the data)
    python experiments/pack_for_colab.py --e22c (cuas_data_e22c.zip: the pasted-drone tiles of build_e22c.py, ~0.1 GB)
    python experiments/pack_for_colab.py --e23 (cuas_data_e23.zip: the E23 frame stacks of build_e23.py)
    python experiments/pack_for_colab.py --e39 (cuas_data_e39.zip: DUT train tiles of build_e39.py)

Writes C:\\cuas_data\\colab\\cuas_data_e14.zip (paths relative to C:\\cuas_data, so on Colab they unpack under
any CUAS_DATA root) and C:\\cuas_data\\colab\\cuas_code.zip (cuas/ + experiments/*.py). Images are already
JPEG, so the zip stores them without recompressing. On Colab, paste experiments/colab_run_all.py into one
cell (E14 train + evaluate, then the M2 detector runs over the Halmstad videos); for E18, experiments/colab_e18.py.
"""
import sys
import zipfile
from pathlib import Path

DATA = Path(r"C:\cuas_data")
REPO = Path(__file__).resolve().parents[1]
OUT = DATA / "colab"
FOLDERS = [  # training (data_e14.yaml) + evaluation sets used by e12_eval.py
    "halmstad_yolo", "e12", "halmstad_zoom_train", "e14/antiuav_hardneg_tiles",
    "antiuav_val", "antiuav_test", "antiuav_zoom_val", "antiuav_zoom_test", "halmstad_zoom",
    "dut_test",                                           # E15: never-trained-on dataset
]
FILES = ["e14/data_e14.yaml",
         "runs/E12_multidata_yolo26s/weights/best.pt",    # M2 detector run + E15 on Colab
         "runs/E12_multidata_yolo26s/e12_eval.json"]      # E12's operating threshold, read by e15_dut.py


def pack_e22c():
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "cuas_data_e22c.zip", "w", zipfile.ZIP_STORED) as z:
        z.write(DATA / "e22c" / "data_e22c.yaml", "e22c/data_e22c.yaml")
        files = [p for p in sorted((DATA / "e22c" / "paste_tiles").rglob("*")) if p.is_file()]
        for p in files:
            z.write(p, p.relative_to(DATA).as_posix())
    print(f"{OUT / 'cuas_data_e22c.zip'}: {(OUT / 'cuas_data_e22c.zip').stat().st_size / 1e9:.3f} GB, {len(files)} files")


def pack_e23(k=3):
    """cuas_data_e23.zip: the E23 frame stacks (build_e23.py); Colab derives the E23ctl set from them."""
    OUT.mkdir(exist_ok=True)
    root = DATA / f"e23_k{k}"
    with zipfile.ZipFile(OUT / "cuas_data_e23.zip", "w", zipfile.ZIP_STORED) as z:
        files = [p for p in sorted(root.rglob("*")) if p.is_file() and p.suffix.lower() in {".jpg", ".txt", ".yaml", ".json"}]
        for p in files:
            z.write(p, p.relative_to(DATA).as_posix())
    print(f"{OUT / 'cuas_data_e23.zip'}: {(OUT / 'cuas_data_e23.zip').stat().st_size / 1e9:.3f} GB, {len(files)} files")


def pack_e39():
    """cuas_data_e39.zip: DUT train tiles and data_e39.yaml (build_e39.py); needs the E22c zip on Drive too."""
    OUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "cuas_data_e39.zip", "w", zipfile.ZIP_STORED) as z:
        files = [p for p in sorted((DATA / "e39").rglob("*")) if p.is_file()]
        for p in files:
            z.write(p, p.relative_to(DATA).as_posix())
    print(f"{OUT / 'cuas_data_e39.zip'}: {(OUT / 'cuas_data_e39.zip').stat().st_size / 1e9:.3f} GB, {len(files)} files")


def main(code_only=False):
    OUT.mkdir(exist_ok=True)
    n = 0
    if not code_only:
        with zipfile.ZipFile(OUT / "cuas_data_e14.zip", "w", zipfile.ZIP_STORED) as z:
            for rel in FILES:
                z.write(DATA / rel, rel)
            for rel in FOLDERS:
                for p in sorted((DATA / rel).rglob("*")):
                    if p.is_file() and p.suffix.lower() in {".jpg", ".png", ".txt", ".yaml", ".json"}:
                        z.write(p, p.relative_to(DATA).as_posix())
                        n += 1
    code = (sorted((REPO / "cuas").rglob("*.py")) + sorted((REPO / "experiments").glob("*.py"))
            + [REPO / "tests" / "test_arch.py"])                 # E18 checks its add-ons on Colab's torch first
    with zipfile.ZipFile(OUT / "cuas_code.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in code:
            if "__pycache__" not in p.parts:
                z.write(p, p.relative_to(REPO).as_posix())
    for f in (("cuas_code.zip",) if code_only else ("cuas_data_e14.zip", "cuas_code.zip")):
        print(f"{OUT / f}: {(OUT / f).stat().st_size / 1e9:.3f} GB")
    if not code_only:
        print(f"{n} data files")


if __name__ == "__main__":
    if "--e22c" in sys.argv:
        pack_e22c()
    elif "--e23" in sys.argv:
        pack_e23()
    elif "--e39" in sys.argv:
        pack_e39()
    else:
        main(code_only="--code" in sys.argv)
