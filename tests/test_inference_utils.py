from dataclasses import dataclass

import pandas as pd

from src.inference_utils import infer_dataframe


@dataclass
class FakeResponse:
    text: str
    latency_seconds: float = 0.01
    input_tokens: int = 10
    output_tokens: int = 5
    attempts: int = 1
    status_code: int = 200


class FakeClient:
    def __init__(self, response_text):
        self.response_text = response_text

    def annotate(self, system_prompt, user_prompt, temperature=0.0):
        return FakeResponse(self.response_text)


def test_infer_dataframe_valid_response():
    df = pd.DataFrame([{
        "segment_id": "r1::s0",
        "review_id": "r1",
        "segment_text": "It keeps my skin hydrated.",
    }])

    response = """
    {"results":[{"segment_id":"r1::s0","aspects":[
      {"aspect_id":"hydration_dryness","sentiment":"positive",
       "evidence":"keeps my skin hydrated"}
    ]}]}
    """

    status, pairs, requests = infer_dataframe(
        df,
        client=FakeClient(response),
        system_prompt="test",
        allowed_aspects={"hydration_dryness"},
        batch_size=1,
        run_name="unit",
        prompt_version="test",
    )

    assert len(status) == 1
    assert bool(status.loc[0, "response_parse_valid"]) is True
    assert bool(status.loc[0, "segment_valid"]) is True
    assert len(pairs) == 1
    assert pairs.loc[0, "aspect_id"] == "hydration_dryness"
    assert pairs.loc[0, "sentiment"] == "positive"
    assert len(requests) == 1


def test_infer_dataframe_handles_invalid_json_without_crashing():
    df = pd.DataFrame([{
        "segment_id": "r1::s0",
        "review_id": "r1",
        "segment_text": "It keeps my skin hydrated.",
    }])

    status, pairs, requests = infer_dataframe(
        df,
        client=FakeClient("not valid json"),
        system_prompt="test",
        allowed_aspects={"hydration_dryness"},
        batch_size=1,
    )

    assert bool(status.loc[0, "response_parse_valid"]) is False
    assert bool(status.loc[0, "segment_valid"]) is False
    assert pairs.empty
    assert requests.loc[0, "parse_error"] != ""
