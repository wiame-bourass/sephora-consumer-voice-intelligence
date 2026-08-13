from pathlib import Path
import json
import pandas as pd

EXPECTED_REVIEW_FILES = [
    "reviews_0-250.csv",
    "reviews_250-500.csv",
    "reviews_500-750.csv",
    "reviews_750-1250.csv",
    "reviews_1250-end.csv",
]
EXPECTED_PRODUCT_FILE = "product_info.csv"


def project_root(start=None):
    p = Path(start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "config" / "project_config.json").exists():
            return candidate
    raise FileNotFoundError("Project root not found. Open Jupyter from the project folder.")


def load_config(root=None):
    root = Path(root or project_root())
    return json.loads((root / "config" / "project_config.json").read_text(encoding="utf-8"))


def check_raw_files(root=None):
    root = Path(root or project_root())
    raw = root / "data" / "raw"
    expected = [EXPECTED_PRODUCT_FILE] + EXPECTED_REVIEW_FILES
    rows = []
    for name in expected:
        p = raw / name
        rows.append({
            "file": name,
            "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 1024**2, 2) if p.exists() else None,
        })
    return pd.DataFrame(rows)


def read_products(root=None, usecols=None):
    root = Path(root or project_root())
    path = root / "data" / "raw" / EXPECTED_PRODUCT_FILE
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def read_all_reviews(root=None, usecols=None, nrows_per_file=None):
    root = Path(root or project_root())
    raw = root / "data" / "raw"
    frames = []
    for name in EXPECTED_REVIEW_FILES:
        path = raw / name
        df = pd.read_csv(path, usecols=usecols, nrows=nrows_per_file, low_memory=False)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df["source_file"] = name
        df["source_row"] = df.index.astype("int64")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def iter_review_chunks(root=None, chunksize=100_000, usecols=None):
    root = Path(root or project_root())
    raw = root / "data" / "raw"
    for name in EXPECTED_REVIEW_FILES:
        path = raw / name
        offset = 0
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            if "Unnamed: 0" in chunk.columns:
                chunk = chunk.drop(columns=["Unnamed: 0"])
            chunk = chunk.reset_index(drop=True)
            chunk["source_file"] = name
            chunk["source_row"] = range(offset, offset + len(chunk))
            offset += len(chunk)
            yield name, chunk


def list_parquet_parts(directory):
    directory = Path(directory)
    return sorted(directory.glob("*.parquet"))
