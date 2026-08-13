import math

import pandas as pd

from src.business_utils import aggregate_aspect_sentiment, wilson_interval


def test_wilson_interval_bounds():
    low, high = wilson_interval(20, 100)

    assert 0 <= low <= 0.20 <= high <= 1


def test_wilson_interval_empty_sample_returns_nan():
    low, high = wilson_interval(0, 0)

    assert math.isnan(low)
    assert math.isnan(high)


def test_aggregate_aspect_sentiment_counts_and_rates():
    pairs = pd.DataFrame([
        {"segment_id": "s1", "review_id": "r1", "category": "Cleansers",
         "aspect_id": "fragrance_smell", "sentiment": "negative"},
        {"segment_id": "s2", "review_id": "r2", "category": "Cleansers",
         "aspect_id": "fragrance_smell", "sentiment": "positive"},
        {"segment_id": "s3", "review_id": "r3", "category": "Cleansers",
         "aspect_id": "fragrance_smell", "sentiment": "negative"},
    ])

    out = aggregate_aspect_sentiment(
        pairs,
        group_cols=["category"],
        support_threshold=3,
    )

    row = out.iloc[0]
    assert row["n_mentions"] == 3
    assert row["n_segments"] == 3
    assert row["n_reviews"] == 3
    assert row["negative_rate"] == 2 / 3
    assert row["positive_rate"] == 1 / 3
    assert bool(row["enough_support"]) is True
