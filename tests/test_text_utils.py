from src.text_utils import (
    normalize_whitespace,
    normalize_for_dedup,
    contains_contrast,
)


def test_normalize_whitespace():
    assert normalize_whitespace("hello   world") == "hello world"


def test_normalize_for_dedup():
    assert normalize_for_dedup("Great Product!!!") == "great product"


def test_contains_contrast():
    assert contains_contrast("I like it but the smell is strong")
    assert not contains_contrast("I really like this product")
