"""OpenAI-compatible client for the CARDEA vLLM endpoint."""

import os

from openai import AsyncOpenAI

MODEL = "cardea"
_client = None


def get_client():
    global _client
    if _client is None:
        base_url = os.environ.get("CARDEA_BASE_URL")
        if not base_url:
            raise RuntimeError(
                "set CARDEA_BASE_URL to your vLLM endpoint, "
                "e.g. http://localhost:8000/v1"
            )
        _client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    return _client


async def stream_text(
    messages,
    *,
    thinking=True,
    temperature=0.5,
    top_p=0.95,
    max_tokens=4096,
):
    """Yield raw model content deltas."""
    response = await get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
