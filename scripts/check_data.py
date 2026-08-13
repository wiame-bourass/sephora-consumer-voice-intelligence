from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.io_utils import check_raw_files

status = check_raw_files(ROOT)
print(status.to_string(index=False))
if not status["exists"].all():
    raise SystemExit("\nMissing raw files. Copy all 6 Sephora CSVs into data/raw/.")
print("\nOK — all expected raw files are present.")
