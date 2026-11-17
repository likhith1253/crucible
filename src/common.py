"""Shared paths and constants. Raw inputs live in data/, intermediate files in work/ (not tracked), outputs in results/."""
import os, json, hashlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = REPO
DATA = os.environ.get("RR_DATA", os.path.join(REPO, "data"))
WORK = os.environ.get("RR_WORK", os.path.join(REPO, "work"))
OUT = os.environ.get("RR_OUT", os.path.join(REPO, "results"))

for _d in (WORK, OUT):
    os.makedirs(_d, exist_ok=True)

SEED = 20260718
BOWLER_CREDITED = {"Caught", "Bowled", "LBW", "Stump", "Hit Wicket"}
CLASSES = ["0", "1", "2", "3", "4", "6", "W", "R", "X"]  # outcome classes, see s2_models.py


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, default=str)


def fingerprint(extra=None):
    """Hash of the pipeline code + raw inputs (+ optional config). Any change invalidates cached fits."""
    import glob
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(CODE, "src", "*.py"))):
        h.update(os.path.basename(f).encode()); h.update(open(f, "rb").read())
    for f in sorted(os.listdir(DATA)):
        if f.endswith(".csv"): h.update(f.encode()); h.update(sha256(os.path.join(DATA, f)).encode())
    if extra is not None: h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]

def atomic_pickle(obj, path):
    import pickle
    tmp = path + ".tmp"
    with open(tmp, "wb") as f: pickle.dump(obj, f)
    os.replace(tmp, path)
