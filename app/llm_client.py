"""
llm_client.py — LLM client for HuggingFace Inference API (OpenAI-compatible router).
"""

import time
from typing import Optional

import requests

from app.config import HF_TOKEN, HF_API_URL, HF_MODEL_ID


# ── Custom exception ───────────────────────────────────────────────────────────

class LLMError(Exception):
    """Raised when an LLM API call fails."""
    pass


# ── Helper functions ───────────────────────────────────────────────────────────

def _build_headers() -> dict:
    """
    Builds the authorization headers for the HuggingFace API.

    Raises:
        LLMError: If HF_TOKEN is missing from environment.
    """
    if not HF_TOKEN:
        raise LLMError("HF_TOKEN is missing from environment (.env).")

    return {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }


# ── Main LLM call ──────────────────────────────────────────────────────────────

def call_llm(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_retries: int = 2,
    timeout: int = 30,
) -> str:
    """
    Calls the LLM via the HuggingFace router (OpenAI-compatible API)
    and returns the generated text.

    Args:
        prompt: Input prompt string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (lower = more deterministic)
        top_p: Nucleus sampling probability
        max_retries: Number of retry attempts on failure
        timeout: Request timeout in seconds

    Returns:
        Generated text string

    Raises:
        ValueError: If prompt is empty
        LLMError: If all retry attempts fail
    """
    if not prompt or not prompt.strip():
        raise ValueError("Empty or invalid prompt passed to call_llm().")

    headers = _build_headers()
    payload = {
        "model": HF_MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code != 200:
                raise LLMError(
                    f"HuggingFace API error (status {response.status_code}): {response.text}"
                )

            data    = response.json()
            choices = data.get("choices", [])

            if not choices:
                raise LLMError("LLM response contains no 'choices'.")

            text = choices[0].get("message", {}).get("content", "").strip()

            if not text:
                raise LLMError("LLM returned an empty response.")

            return text

        except Exception as e:
            print(f"[LLM] Attempt {attempt}/{max_retries} failed: {e}")
            last_error = e
            time.sleep(1)

    raise LLMError(f"LLM call failed after {max_retries} attempts: {last_error}")