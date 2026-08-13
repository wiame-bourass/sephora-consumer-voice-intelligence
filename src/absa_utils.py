import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

ALLOWED_SENTIMENTS = {"positive", "neutral", "negative"}


def load_taxonomy(root):
    root = Path(root)
    return json.loads((root / "config" / "taxonomy_v1.json").read_text(encoding="utf-8"))


def core_aspect_ids(taxonomy):
    return list(taxonomy["core_aspects"].keys())


def taxonomy_table(taxonomy):
    rows = []
    for aspect_id, meta in taxonomy["core_aspects"].items():
        rows.append({
            "aspect_id": aspect_id,
            "display_name": meta.get("display_name", aspect_id),
            "description": meta.get("description", ""),
        })
    return pd.DataFrame(rows)


def build_system_prompt(taxonomy, prompt_version="absa_v2_llm_first"):
    aspect_lines = []
    for aspect_id, meta in taxonomy["core_aspects"].items():
        aspect_lines.append(f'- {aspect_id}: {meta.get("description", "")}')
    aspects = "\n".join(aspect_lines)
    return f"""You are an Aspect-Based Sentiment Analysis engine for skincare consumer reviews.
Prompt version: {prompt_version}

Your task is extraction, not summarization.
For each input segment, identify zero, one, or several supported aspects from the CLOSED taxonomy below.
For each detected aspect assign exactly one sentiment: positive, neutral, or negative.
Return an evidence string copied EXACTLY from the input segment.
Do not infer an aspect from product name, brand, rating, skin type metadata, or general world knowledge.
Do not invent evidence. If no taxonomy aspect is explicitly supported by the segment, return an empty aspects list.
Do not return watchlist aspects or any label outside the closed taxonomy.

CLOSED TAXONOMY:
{aspects}

Boundary rules:
- A consumer saying their skin is dry is profile/context, not automatically hydration_dryness. A product hydrating, drying, stripping, or preventing dryness is hydration_dryness.
- Oily/greasy describing the consumer's skin type is not texture_finish. Oily/greasy describing product feel or finish is texture_finish.
- Redness caused by the product is irritation_sensitivity. Reduction of pre-existing redness is efficacy_results unless the text clearly describes tolerability/irritation.
- An ingredient name alone is not a core aspect.
- One segment may contain multiple aspects with different sentiments.

Return ONLY valid JSON with this exact top-level shape:
{{"results":[{{"segment_id":"...","aspects":[{{"aspect_id":"...","sentiment":"positive|neutral|negative","evidence":"exact substring"}}]}}]}}
"""


def build_batch_user_prompt(items):
    payload = [{"segment_id": str(x["segment_id"]), "text": str(x["segment_text"])} for x in items]
    return "Analyze these segments:\n" + json.dumps(payload, ensure_ascii=False)


def extract_json_object(text):
    if text is None:
        raise ValueError("Empty LLM response")
    text = str(text).strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end+1])
        raise


def _norm_for_evidence(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def evidence_is_substring(evidence, segment_text):
    e = _norm_for_evidence(evidence)
    t = _norm_for_evidence(segment_text)
    return bool(e) and e in t


def validate_payload(payload, expected_items, allowed_aspects):
    expected = {str(x["segment_id"]): str(x["segment_text"]) for x in expected_items}
    issues = []
    clean_results = []
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        return [], ["Top-level JSON must contain a list field named 'results'."]

    seen_ids = set()
    for result in payload["results"]:
        if not isinstance(result, dict):
            issues.append("Non-object result entry.")
            continue
        sid = str(result.get("segment_id", ""))
        if sid not in expected:
            issues.append(f"Unknown segment_id returned: {sid}")
            continue
        if sid in seen_ids:
            issues.append(f"Duplicate segment_id returned: {sid}")
            continue
        seen_ids.add(sid)
        raw_aspects = result.get("aspects", [])
        if not isinstance(raw_aspects, list):
            issues.append(f"{sid}: aspects must be a list")
            raw_aspects = []
        cleaned = []
        seen_aspects = set()
        for a in raw_aspects:
            if not isinstance(a, dict):
                issues.append(f"{sid}: non-object aspect entry")
                continue
            aspect_id = str(a.get("aspect_id", "")).strip()
            sentiment = str(a.get("sentiment", "")).strip().lower()
            evidence = str(a.get("evidence", "")).strip()
            if aspect_id not in allowed_aspects:
                issues.append(f"{sid}: invalid aspect '{aspect_id}'")
                continue
            if sentiment not in ALLOWED_SENTIMENTS:
                issues.append(f"{sid}: invalid sentiment '{sentiment}' for {aspect_id}")
                continue
            if aspect_id in seen_aspects:
                issues.append(f"{sid}: duplicate aspect '{aspect_id}'")
                continue
            seen_aspects.add(aspect_id)
            evidence_ok = evidence_is_substring(evidence, expected[sid])
            if not evidence_ok:
                issues.append(f"{sid}: evidence is not an exact substring for {aspect_id}")
            cleaned.append({
                "aspect_id": aspect_id,
                "sentiment": sentiment,
                "evidence": evidence,
                "evidence_exact_match": bool(evidence_ok),
            })
        clean_results.append({"segment_id": sid, "aspects": cleaned})

    missing = sorted(set(expected) - seen_ids)
    for sid in missing:
        issues.append(f"Missing segment_id: {sid}")
        clean_results.append({"segment_id": sid, "aspects": []})
    return clean_results, issues


def flatten_results(clean_results, metadata_df):
    meta = metadata_df.copy()
    meta["segment_id"] = meta["segment_id"].astype(str)
    meta = meta.set_index("segment_id", drop=False)
    status_rows, pair_rows = [], []
    for r in clean_results:
        sid = str(r["segment_id"])
        base = meta.loc[sid].to_dict() if sid in meta.index else {"segment_id": sid}
        aspects = r.get("aspects", [])
        status_rows.append({**base, "n_aspects": len(aspects)})
        for a in aspects:
            pair_rows.append({**base, **a})
    return pd.DataFrame(status_rows), pd.DataFrame(pair_rows)


def approximate_tokens(texts: Iterable[str]):
    # Provider-agnostic rough estimate. It is a planning aid, not billing truth.
    chars = sum(len(str(t)) for t in texts)
    return int(round(chars / 4.0))
