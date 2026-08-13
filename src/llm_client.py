import os
import time
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


@dataclass
class LLMResponse:
    text: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    status_code: int | None = None
    attempts: int = 1


class ChatCompletionsHTTPClient:
    """Minimal provider-agnostic client for chat-completions-compatible HTTP endpoints.

    The FULL endpoint URL is provided through LLM_ENDPOINT_URL, so this module does
    not assume a particular vendor base URL.
    """

    def __init__(self, endpoint_url, api_key, model, auth_scheme="Bearer", timeout=120, max_retries=3, backoff=2.0):
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.model = model
        self.auth_scheme = auth_scheme
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    @classmethod
    def from_env(cls, root, config):
        load_dotenv(os.path.join(root, ".env"))
        endpoint = os.getenv("LLM_ENDPOINT_URL", "").strip()
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "").strip()
        auth_scheme = os.getenv("LLM_AUTH_SCHEME", "Bearer").strip() or "Bearer"
        missing = [k for k, v in {
            "LLM_ENDPOINT_URL": endpoint,
            "LLM_API_KEY": api_key,
            "LLM_MODEL": model,
        }.items() if not v]
        if missing:
            raise RuntimeError("Missing environment variables: " + ", ".join(missing) + ". Copy .env.example to .env first.")
        llm_cfg = config["llm"]
        return cls(
            endpoint_url=endpoint,
            api_key=api_key,
            model=model,
            auth_scheme=auth_scheme,
            timeout=llm_cfg.get("request_timeout_seconds", 120),
            max_retries=llm_cfg.get("max_retries", 3),
            backoff=llm_cfg.get("retry_backoff_seconds", 2.0),
        )

    def annotate(self, system_prompt, user_prompt, temperature=0.0):
        headers = {
            "Authorization": f"{self.auth_scheme} {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "reasoning_effort": "low",
            "max_tokens": 1024,
            "stream": False,
        }

        last_error = None

        for attempt in range(1, self.max_retries + 2):
            t0 = time.perf_counter()

            try:
                resp = requests.post(
                    self.endpoint_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                latency = time.perf_counter() - t0

                # Afficher LA VRAIE erreur NVIDIA
                if not resp.ok:
                    print("\n===== NVIDIA ERROR =====")
                    print("STATUS:", resp.status_code)
                    print("BODY:", resp.text)
                    print("MODEL:", self.model)
                    print("SYSTEM PROMPT CHARS:", len(system_prompt))
                    print("USER PROMPT CHARS:", len(user_prompt))
                    print("========================\n")

                # Retry seulement pour rate limit / erreur serveur
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text[:1000]}"
                    )

                    if attempt <= self.max_retries:
                        time.sleep(self.backoff * (2 ** (attempt - 1)))
                        continue

                # Les 400/401/403/etc. ne servent à rien à retry
                if 400 <= resp.status_code < 500:
                    raise RuntimeError(
                        f"HTTP {resp.status_code}: {resp.text}"
                    )

                resp.raise_for_status()

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {}) or {}

                return LLMResponse(
                    text=content,
                    latency_seconds=latency,
                    input_tokens=usage.get("prompt_tokens")
                    or usage.get("input_tokens"),
                    output_tokens=usage.get("completion_tokens")
                    or usage.get("output_tokens"),
                    status_code=resp.status_code,
                    attempts=attempt,
                )

            except RuntimeError:
                raise

            except Exception as exc:
                last_error = exc

                if attempt <= self.max_retries:
                    time.sleep(self.backoff * (2 ** (attempt - 1)))
                    continue

                raise RuntimeError(
                    f"LLM request failed after {attempt} attempts: {last_error}"
                ) from last_error