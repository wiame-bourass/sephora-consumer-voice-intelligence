import json
import time
from pathlib import Path

import pandas as pd

from .absa_utils import (
    build_batch_user_prompt, extract_json_object, validate_payload, flatten_results,
)

PAIR_EXTRA_COLUMNS = ["aspect_id", "sentiment", "evidence", "evidence_exact_match",
                      "run_name", "batch_id", "prompt_version"]


def _issue_count_for_segment(issues, segment_id, parse_error=""):
    if parse_error:
        return 1
    sid = str(segment_id)
    return sum(1 for issue in issues if sid in str(issue))


def infer_dataframe(df, client, system_prompt, allowed_aspects, batch_size=20, temperature=0.0,
                    raw_dir=None, run_name="run", prompt_version="unknown", batch_id_offset=0):
    if df.empty:
        empty_pairs = pd.DataFrame(columns=list(df.columns) + PAIR_EXTRA_COLUMNS)
        return pd.DataFrame(), empty_pairs, pd.DataFrame()
    raw_dir = Path(raw_dir) if raw_dir else None
    if raw_dir:
        raw_dir.mkdir(parents=True, exist_ok=True)

    all_status, all_pairs, request_rows = [], [], []
    records = df.to_dict("records")
    for start in range(0, len(records), batch_size):
        items = records[start:start + batch_size]
        local_batch_id = start // batch_size
        batch_id = batch_id_offset + local_batch_id
        user_prompt = build_batch_user_prompt(items)
        t0 = time.perf_counter()
        response = client.annotate(system_prompt, user_prompt, temperature=temperature)
        wall = time.perf_counter() - t0
        parse_error = ""
        issues = []
        try:
            payload = extract_json_object(response.text)
            clean_results, issues = validate_payload(payload, items, allowed_aspects)
        except Exception as exc:
            parse_error = repr(exc)
            clean_results = [{"segment_id": str(x["segment_id"]), "aspects": []} for x in items]
            issues = ["PARSE_ERROR: " + parse_error]

        meta_df = pd.DataFrame(items)
        status, pairs = flatten_results(clean_results, meta_df)
        status["run_name"] = run_name
        status["batch_id"] = batch_id
        status["prompt_version"] = prompt_version
        status["response_parse_valid"] = not bool(parse_error)
        status["issues_count"] = status["segment_id"].map(
            lambda sid: _issue_count_for_segment(issues, sid, parse_error=parse_error)
        )
        status["segment_valid"] = status["response_parse_valid"] & status["issues_count"].eq(0)
        status["batch_issues_count"] = len(issues)
        status["issues_json"] = json.dumps(issues, ensure_ascii=False)
        if not pairs.empty:
            pairs["run_name"] = run_name
            pairs["batch_id"] = batch_id
            pairs["prompt_version"] = prompt_version
        else:
            pairs = pd.DataFrame(columns=list(meta_df.columns) + PAIR_EXTRA_COLUMNS)
        all_status.append(status)
        all_pairs.append(pairs)

        request_rows.append({
            "run_name": run_name,
            "batch_id": batch_id,
            "n_segments": len(items),
            "latency_seconds": response.latency_seconds,
            "wall_seconds": wall,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "attempts": response.attempts,
            "status_code": response.status_code,
            "parse_error": parse_error,
            "batch_issues_count": len(issues),
        })
        if raw_dir:
            raw = {
                "run_name": run_name,
                "batch_id": batch_id,
                "segment_ids": [str(x["segment_id"]) for x in items],
                "raw_text": response.text,
                "issues": issues,
                "parse_error": parse_error,
            }
            (raw_dir / f"{run_name}_batch_{batch_id:06d}.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    status_df = pd.concat(all_status, ignore_index=True) if all_status else pd.DataFrame()
    pairs_df = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame(columns=list(df.columns)+PAIR_EXTRA_COLUMNS)
    requests_df = pd.DataFrame(request_rows)
    return status_df, pairs_df, requests_df


def infer_dataframe_resumable(df, client, system_prompt, allowed_aspects, batch_size=20,
                              checkpoint_batches=50, temperature=0.0, checkpoint_dir=None,
                              raw_dir=None, run_name="run", prompt_version="unknown"):
    """Checkpoint a long dataframe every N API batches.

    Rerunning skips completed blocks, so an interruption does not force replaying an
    entire 100k-row parquet part.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir) if raw_dir else None
    block_size = batch_size * checkpoint_batches
    block_status_paths, block_pair_paths, block_request_paths = [], [], []

    for block_start in range(0, len(df), block_size):
        block_id = block_start // block_size
        st_path = checkpoint_dir / f"status_block_{block_id:05d}.parquet"
        pr_path = checkpoint_dir / f"pairs_block_{block_id:05d}.parquet"
        rq_path = checkpoint_dir / f"requests_block_{block_id:05d}.csv"
        block_status_paths.append(st_path); block_pair_paths.append(pr_path); block_request_paths.append(rq_path)
        if st_path.exists() and pr_path.exists() and rq_path.exists():
            print(f"  SKIP checkpoint block {block_id:05d}")
            continue
        block = df.iloc[block_start:block_start + block_size].copy()
        batch_offset = block_start // batch_size
        st, pr, rq = infer_dataframe(
            block, client, system_prompt, allowed_aspects, batch_size=batch_size,
            temperature=temperature,
            raw_dir=(raw_dir / f"block_{block_id:05d}") if raw_dir else None,
            run_name=run_name, prompt_version=prompt_version,
            batch_id_offset=batch_offset,
        )
        st.to_parquet(st_path, index=False)
        pr.to_parquet(pr_path, index=False)
        rq.to_csv(rq_path, index=False)
        print(f"  wrote checkpoint block {block_id:05d}: {len(st):,} segments")

    statuses = [pd.read_parquet(x) for x in block_status_paths if x.exists()]
    pairs = [pd.read_parquet(x) for x in block_pair_paths if x.exists()]
    requests = [pd.read_csv(x) for x in block_request_paths if x.exists()]
    return (
        pd.concat(statuses, ignore_index=True) if statuses else pd.DataFrame(),
        pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame(columns=list(df.columns)+PAIR_EXTRA_COLUMNS),
        pd.concat(requests, ignore_index=True) if requests else pd.DataFrame(),
    )
