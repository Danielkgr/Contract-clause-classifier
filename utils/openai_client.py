"""
OpenAI, or any server with an OpenAI-compatible chat API, kept for comparison.

Requests carry the same system prompt and excerpt as the Claude arm and ask
for a JSON object.  Claude is never reached through this module.
"""

import time
from collections.abc import Sequence

import openai

from utils.llm_client import (
    Completion,
    FatalLLMError,
    LLMCallError,
    LLMClient,
    TransientLLMError,
    Usage,
)
from utils.prompts import system_prompt, user_message

# OpenAI's finish reasons in the harness's terms
STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}


def _retry_after(exc: openai.APIStatusError) -> float | None:
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None


class OpenAIClient(LLMClient):
    """Asks an OpenAI chat model which clause types appear in an excerpt."""

    provider = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tokens: int = 2048,
        temperature: float | None = None,
        timeout: float = 120.0,
        http_client=None,
        **client_options,
    ):
        if model.startswith("claude-"):
            raise ValueError("Claude models are reached through the anthropic provider")
        super().__init__(model, max_tokens, **client_options)
        self.base_url = base_url
        self.temperature = temperature
        try:
            # LLMClient retries transient errors itself, so the SDK's are off
            self._client = openai.OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=0,
                http_client=http_client,
            )
        except openai.OpenAIError as exc:
            raise ValueError(f"Could not set up the OpenAI client: {exc}") from exc

    def cache_settings(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "base_url": self.base_url,
        }

    def request(self, excerpt: str, clause_types: Sequence[str]) -> dict:
        """The chat completion parameters for one excerpt."""
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt(clause_types)},
                {"role": "user", "content": user_message(excerpt)},
            ],
            # JSON mode keeps the answer parseable.  The shared parser then
            # checks every clause type it names.
            "response_format": {"type": "json_object"},
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def send(self, excerpt: str, clause_types: Sequence[str]) -> Completion:
        start = time.perf_counter()
        try:
            response = self._client.chat.completions.create(**self.request(excerpt, clause_types))
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise FatalLLMError(f"OpenAI rejected the credentials: {exc.message}") from exc
        except openai.NotFoundError as exc:
            raise FatalLLMError(f"model {self.model} is not available: {exc.message}") from exc
        except openai.RateLimitError as exc:
            raise TransientLLMError(f"rate limited: {exc.message}", _retry_after(exc)) from exc
        except (openai.InternalServerError, openai.ConflictError) as exc:
            raise TransientLLMError(
                f"server error {exc.status_code}: {exc.message}", _retry_after(exc)
            ) from exc
        except openai.APIStatusError as exc:
            if exc.status_code == 408:
                raise TransientLLMError(f"request timeout: {exc.message}") from exc
            raise LLMCallError(f"request rejected with {exc.status_code}: {exc.message}") from exc
        except openai.APIConnectionError as exc:  # includes APITimeoutError
            raise TransientLLMError(f"connection failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        stop_reason = STOP_REASONS.get(choice.finish_reason, choice.finish_reason or "")
        refusal = choice.message.refusal
        if refusal:
            stop_reason = "refusal"
        usage = response.usage
        return Completion(
            text=(choice.message.content or "") if stop_reason == "end_turn" else "",
            stop_reason=stop_reason,
            usage=Usage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            latency_ms=latency_ms,
            refusal=refusal or None,
        )
