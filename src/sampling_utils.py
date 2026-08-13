from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def list_segment_parts(root):
    return sorted((Path(root) / "data" / "interim" / "segment_parts").glob("*.parquet"))


def sample_segments(root, n=200, seed=42, min_words=5, columns=None):
    """Uniform-ish streaming sample without loading every segment into one dataframe.

    Each row gets a random key; the n smallest keys observed so far are kept.
    """
    parts = list_segment_parts(root)
    if not parts:
        raise FileNotFoundError("No segment parquet parts. Execute notebook 02 first.")
    rng = np.random.default_rng(seed)
    keep = None
    columns = columns or [
        "segment_id", "review_id", "product_id", "product_name_catalog", "brand_name_catalog",
        "skin_type", "rating", "secondary_category", "tertiary_category", "segment_text",
        "segment_word_count", "has_contrast_marker"
    ]
    for part in parts:
        available = set(pq.read_schema(part).names)
        use = [c for c in columns if c in available]
        df = pd.read_parquet(part, columns=use)
        if "segment_word_count" in df.columns:
            df = df[df["segment_word_count"] >= min_words].copy()
        if df.empty:
            continue
        df["_sample_key"] = rng.random(len(df))
        if keep is None:
            keep = df.nsmallest(min(n, len(df)), "_sample_key")
        else:
            keep = pd.concat([keep, df], ignore_index=True).nsmallest(n, "_sample_key")
    if keep is None:
        return pd.DataFrame()
    return keep.drop(columns=["_sample_key"]).reset_index(drop=True)
