import json

from src.absa_utils import (
    extract_json_object,
    evidence_is_substring,
    validate_payload,
)


ALLOWED = {"hydration_dryness", "fragrance_smell"}


def test_extract_json_object_plain_json():
    payload = extract_json_object('{"results": []}')
    assert payload == {"results": []}


def test_extract_json_object_from_code_fence():
    payload = extract_json_object('```json\n{"results": []}\n```')
    assert payload == {"results": []}


def test_evidence_is_substring_normalizes_case_and_spaces():
    assert evidence_is_substring(
        "Keeps   my skin hydrated",
        "This cream keeps my skin hydrated all day.",
    )


def test_validate_payload_accepts_valid_result():
    items = [{"segment_id": "r1::s0", "segment_text": "It keeps my skin hydrated."}]
    payload = {
        "results": [{
            "segment_id": "r1::s0",
            "aspects": [{
                "aspect_id": "hydration_dryness",
                "sentiment": "positive",
                "evidence": "keeps my skin hydrated",
            }],
        }]
    }

    clean, issues = validate_payload(payload, items, ALLOWED)

    assert issues == []
    assert clean[0]["aspects"][0]["aspect_id"] == "hydration_dryness"
    assert clean[0]["aspects"][0]["evidence_exact_match"] is True


def test_validate_payload_rejects_unknown_aspect():
    items = [{"segment_id": "r1::s0", "segment_text": "Nice color."}]
    payload = {
        "results": [{
            "segment_id": "r1::s0",
            "aspects": [{
                "aspect_id": "color",
                "sentiment": "positive",
                "evidence": "Nice color",
            }],
        }]
    }

    clean, issues = validate_payload(payload, items, ALLOWED)

    assert clean[0]["aspects"] == []
    assert any("invalid aspect" in issue for issue in issues)


def test_validate_payload_rejects_invalid_sentiment():
    items = [{"segment_id": "r1::s0", "segment_text": "The smell is strong."}]
    payload = {
        "results": [{
            "segment_id": "r1::s0",
            "aspects": [{
                "aspect_id": "fragrance_smell",
                "sentiment": "mixed",
                "evidence": "smell is strong",
            }],
        }]
    }

    clean, issues = validate_payload(payload, items, ALLOWED)

    assert clean[0]["aspects"] == []
    assert any("invalid sentiment" in issue for issue in issues)


def test_validate_payload_flags_non_exact_evidence():
    items = [{"segment_id": "r1::s0", "segment_text": "The smell is strong."}]
    payload = {
        "results": [{
            "segment_id": "r1::s0",
            "aspects": [{
                "aspect_id": "fragrance_smell",
                "sentiment": "negative",
                "evidence": "awful fragrance",
            }],
        }]
    }

    clean, issues = validate_payload(payload, items, ALLOWED)

    assert clean[0]["aspects"][0]["evidence_exact_match"] is False
    assert any("evidence is not an exact substring" in issue for issue in issues)


def test_validate_payload_adds_missing_segment():
    items = [
        {"segment_id": "r1::s0", "segment_text": "Hydrating."},
        {"segment_id": "r2::s0", "segment_text": "Strong smell."},
    ]
    payload = {
        "results": [{
            "segment_id": "r1::s0",
            "aspects": [],
        }]
    }

    clean, issues = validate_payload(payload, items, ALLOWED)

    ids = {row["segment_id"] for row in clean}
    assert ids == {"r1::s0", "r2::s0"}
    assert any("Missing segment_id: r2::s0" in issue for issue in issues)
