from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sampling_utils import sample_segments

smoke = sample_segments(
    ROOT,
    n=200,
    seed=42,
    min_words=5,
)

out = (
    ROOT
    / "data"
    / "interim"
    / "smoke_200.parquet"
)

smoke.to_parquet(
    out,
    index=False,
)

print("Created:", out)
print("Rows:", len(smoke))