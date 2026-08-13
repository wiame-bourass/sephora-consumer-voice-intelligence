from src.absa_utils import evidence_is_substring


def test_evidence_is_substring():
    text = "This cream keeps my skin hydrated."

    assert evidence_is_substring(
        "keeps my skin hydrated",
        text,
    )


def test_evidence_not_substring():
    text = "This cream feels lightweight."

    assert not evidence_is_substring(
        "keeps my skin hydrated",
        text,
    )
