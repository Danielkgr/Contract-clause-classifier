"""
Claude through the official anthropic SDK.

This is the only module that calls Anthropic.  It builds one Messages API
request per excerpt and maps the SDK's typed errors onto the harness's
fatal, retryable, and per-call failures.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import anthropic

from utils.llm_client import (
    Completion,
    FatalLLMError,
    LLMCallError,
    LLMClient,
    TransientLLMError,
    Usage,
)
from utils.prompts import answer_schema, system_prompt, user_message

EFFORTS = ("low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class ModelOptions:
    """Which request settings a model accepts."""

    effort: bool  # takes output_config.effort, and thinks adaptively
    temperature: bool  # accepts a temperature


# The models this harness knows how to call.  Opus 5.5, Sonnet 5.5, and Fable
# 5.1 reject temperature, and Opus 5.5 always thinks, so effort is the only
# depth control there.  Haiku 4.5 takes a temperature but no effort setting.
MODELS = {
    "claude-opus-5-5": ModelOptions(effort=True, temperature=False),
    "claude-sonnet-5-5": ModelOptions(effort=True, temperature=False),
    "claude-haiku-4-5": ModelOptions(effort=False, temperature=True),
    "claude-fable-5-1": ModelOptions(effort=True, temperature=False),
}


def _retry_after(exc: anthropic.APIStatusError) -> float | None:
    """The wait in seconds a response asked for in its retry-after header, if any."""
    try:
        return float(exc.response.headers.get("retry-after", ""))
    except ValueError:
        return None


class AnthropicClient(LLMClient):
    """Asks Claude which clause types appear in an excerpt."""

    provider = "anthropic"

    def __init__(
        self,
        model: str = "claude-opus-5-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 2048,
        effort: str = "low",
        temperature: float | None = None,
        timeout: float = 120.0,
        http_client=None,
        **client_options,
    ):
        """
        Args:
            model: one of MODELS
            api_key: the key, or None to let the SDK read ANTHROPIC_API_KEY
            max_tokens: cap on output tokens, thinking included
            effort: thinking depth for models that take it, one of EFFORTS
            temperature: sent only when set, and only to a model that accepts it
            timeout: seconds to wait for one response
            http_client: an httpx2.Client to use instead of the SDK's own
            client_options: cache, retry, and sleep, passed to LLMClient
        """
        if model not in MODELS:
            raise ValueError(
                f"{model} is not one of the Claude models this harness knows: {list(MODELS)}"
            )
        self.options = MODELS[model]
        if temperature is not None and not self.options.temperature:
            raise ValueError(f"{model} rejects temperature, so leave LLM_TEMPERATURE unset")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        super().__init__(model, max_tokens, **client_options)
        self.effort = effort
        self.temperature = temperature
        # LLMClient retries transient errors itself, so the SDK's own retries
        # are off.  Measured latency then covers one attempt, never a backoff.
        self._client = anthropic.Anthropic(
            api_key=api_key, timeout=timeout, max_retries=0, http_client=http_client
        )

    def cache_settings(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "effort": self.effort if self.options.effort else None,
            "temperature": self.temperature,
        }

    def request(self, excerpt: str, clause_types: Sequence[str]) -> dict:
        """The Messages API parameters for one excerpt."""
        output_config: dict = {
            "format": {"type": "json_schema", "schema": answer_schema(clause_types)}
        }
        if self.options.effort:
            output_config["effort"] = self.effort
        params = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The definitions come first and never change between excerpts, so
            # they are marked as the end of the cacheable prefix.  The API caches
            # them only once they reach the model's minimum cacheable length.
            "system": [
                {
                    "type": "text",
                    "text": system_prompt(clause_types),
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message(excerpt)}],
            "output_config": output_config,
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        return params

    def send(self, excerpt: str, clause_types: Sequence[str]) -> Completion:
        params = self.request(excerpt, clause_types)
        # The 1.x SDK takes temperature only in the request body
        extra_body = {"temperature": params.pop("temperature")} if "temperature" in params else None
        start = time.perf_counter()
        try:
            # No server-side fallbacks: this harness measures one named model, and
            # a fallback would let a different model answer without it showing in
            # the results.  A refusal is recorded as its own failed outcome.
            message = self._client.messages.create(**params, extra_body=extra_body)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError) as exc:
            raise FatalLLMError(f"Anthropic rejected the credentials: {exc.message}") from exc
        except anthropic.NotFoundError as exc:
            raise FatalLLMError(f"model {self.model} is not available: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise TransientLLMError(f"rate limited: {exc.message}", _retry_after(exc)) from exc
        except (anthropic.OverloadedError, anthropic.InternalServerError) as exc:
            raise TransientLLMError(
                f"server error {exc.status_code}: {exc.message}", _retry_after(exc)
            ) from exc
        except anthropic.ConflictError as exc:
            raise TransientLLMError(f"conflict: {exc.message}", _retry_after(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 402:
                raise FatalLLMError(f"billing problem: {exc.message}") from exc
            if exc.status_code == 408:
                raise TransientLLMError(f"request timeout: {exc.message}") from exc
            raise LLMCallError(f"request rejected with {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:  # includes APITimeoutError
            raise TransientLLMError(f"connection failed: {exc}") from exc
        latency_ms = (time.perf_counter() - start) * 1000

        usage = message.usage
        text, refusal = "", None
        if message.stop_reason == "refusal":
            # stop_details is read only on a refusal, and may still be null
            details = message.stop_details
            if details is not None:
                refusal = ": ".join(
                    part for part in (details.category, details.explanation) if part
                )
        elif message.stop_reason == "end_turn":
            text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(
            text=text,
            stop_reason=message.stop_reason or "",
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_tokens=usage.cache_read_input_tokens or 0,
                cache_write_tokens=usage.cache_creation_input_tokens or 0,
            ),
            latency_ms=latency_ms,
            refusal=refusal or None,
        )
