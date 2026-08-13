import pandas as pd

from src.text_utils import (
    candidate_aspects,
    compile_aspect_patterns,
    contains_contrast,
    normalize_for_dedup,
    normalize_whitespace,
    sentence_split_series,
)


def test_normalize_whitespace():
    assert normalize_whitespace("hello   world") == "hello world"


def test_normalize_for_dedup_removes_url_and_punctuation():
    text = "Great Product!!! https://example.com"
    assert normalize_for_dedup(text) == "great product"


def test_contains_contrast():
    assert contains_contrast("I like it but the smell is strong")
    assert not contains_contrast("I really like this product")


def test_sentence_split_series_splits_multiple_sentences():
    df = pd.DataFrame({
        "review_id": ["r1"],
        "review_text": ["Great texture. Smells awful! Works well?"],
    })

    out = sentence_split_series(df)

    assert out["segment_text"].tolist() == [
        "Great texture.",
        "Smells awful!",
        "Works well?",
    ]
    assert out["segment_id"].tolist() == ["r1::s0", "r1::s1", "r1::s2"]


def test_sentence_split_does_not_split_on_but():
    df = pd.DataFrame({
        "review_id": ["r1"],
        "review_text": ["Very hydrating but the smell is awful."],
    })

    out = sentence_split_series(df)

    assert len(out) == 1
    assert out.iloc[0]["has_contrast_marker"]


def test_candidate_aspects_from_taxonomy_keywords():
    taxonomy = {
        "core_aspects": {
            "hydration_dryness": {"keywords": ["hydrating", "dry"]},
            "fragrance_smell": {"keywords": ["smell", "fragrance"]},
        }
    }
    patterns = compile_aspect_patterns(taxonomy)

    found = candidate_aspects("Hydrating, but the smell is strong.", patterns)

    assert set(found) == {"hydration_dryness", "fragrance_smell"}
