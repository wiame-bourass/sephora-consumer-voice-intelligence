import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support, f1_score


def automatic_quality_summary(status_df, pairs_df):
    out = {
        "segments": int(len(status_df)),
        "response_parse_valid_rate": float(status_df["response_parse_valid"].mean()) if len(status_df) else np.nan,
        "segment_valid_rate": float(status_df["segment_valid"].mean()) if len(status_df) else np.nan,
        "segments_with_aspect_rate": float((status_df["n_aspects"] > 0).mean()) if len(status_df) else np.nan,
        "mean_aspects_per_segment": float(status_df["n_aspects"].mean()) if len(status_df) else np.nan,
        "schema_issue_segment_rate": float((status_df["issues_count"] > 0).mean()) if len(status_df) else np.nan,
    }
    if len(pairs_df):
        out["evidence_exact_match_rate"] = float(pairs_df["evidence_exact_match"].mean())
    else:
        out["evidence_exact_match_rate"] = np.nan
    return pd.Series(out, name="value").to_frame()


def prediction_signature(pairs_df):
    if pairs_df.empty:
        return {}
    tmp = pairs_df.copy()
    tmp["pair"] = tmp["aspect_id"].astype(str) + "::" + tmp["sentiment"].astype(str)
    return tmp.groupby("segment_id")["pair"].apply(lambda x: tuple(sorted(set(x)))).to_dict()


def agreement_table(status_a, pairs_a, status_b, pairs_b):
    sig_a = prediction_signature(pairs_a)
    sig_b = prediction_signature(pairs_b)
    ids = sorted(set(status_a["segment_id"].astype(str)) | set(status_b["segment_id"].astype(str)))
    rows = []
    for sid in ids:
        a = set(sig_a.get(sid, ()))
        b = set(sig_b.get(sid, ()))
        union = a | b
        jaccard = len(a & b) / len(union) if union else 1.0
        rows.append({"segment_id": sid, "exact_agreement": a == b, "jaccard": jaccard,
                     "run_a": " | ".join(sorted(a)), "run_b": " | ".join(sorted(b))})
    return pd.DataFrame(rows)


