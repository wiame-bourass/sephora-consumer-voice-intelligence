from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
archive = ROOT / "data" / "raw" / "archive.zip"
if not archive.exists():
    raise SystemExit("Put your Kaggle archive zip as data/raw/archive.zip first, or copy the 6 CSV files manually.")
with zipfile.ZipFile(archive) as z:
    z.extractall(RAW)
print("Extracted to", RAW)
