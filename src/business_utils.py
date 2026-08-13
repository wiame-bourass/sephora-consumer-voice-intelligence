import math
import pandas as pd


def wilson_interval(successes, n, z=1.96):
    if n <= 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    margin = z * math.sqrt((phat*(1-phat) + z*z/(4*n))/n) / denom
    return center - margin, center + margin


def aggregate_aspect_sentiment(pairs_df, group_cols, support_threshold=30):
    if pairs_df.empty:
        return pd.DataFrame()
    base = pairs_df.groupby(group_cols + ["aspect_id"]).agg(
        n_mentions=("segment_id", "size"),
        n_segments=("segment_id", "nunique"),
        n_reviews=("review_id", "nunique"),
    ).reset_index()
    sent = (pairs_df.assign(_n=1)
            .pivot_table(index=group_cols+["aspect_id"], columns="sentiment", values="_n", aggfunc="sum", fill_value=0)
            .reset_index())
    out = base.merge(sent, on=group_cols+["aspect_id"], how="left")
    for s in ["positive","neutral","negative"]:
        if s not in out.columns: out[s] = 0
        out[f"{s}_rate"] = out[s] / out["n_mentions"]
    intervals = out.apply(lambda r: wilson_interval(r["negative"], r["n_mentions"]), axis=1)
    out["negative_rate_low"] = [x[0] for x in intervals]
    out["negative_rate_high"] = [x[1] for x in intervals]
    out["enough_support"] = out["n_mentions"] >= support_threshold
    return out
