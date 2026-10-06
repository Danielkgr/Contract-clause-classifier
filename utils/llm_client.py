"""
The provider-neutral part of the zero-shot LLM arm.

A provider module turns one request into a Completion: the answer text, why
the model stopped, the token usage, and the latency.  LLMClient.classify()
reads that into an LLMResponse, which says which clause types the model
found, or why the call failed.  A failed call is never read as "absent".

Claude is reached only through utils/anthropic_client.py, and OpenAI only
through utils/openai_client.py.  make_client() picks one by provider.
"""

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from utils.pricing import cost_usd, price_for
from utils.prompts import parse_answer

logger = logging.getLogger(__name__)

PROVIDERS = ("anthropic", "openai")

# How a call ended.  Every outcome but "ok" is a failed call.
OUTCOMES = ("ok", "refusal", "truncated", "unparseable", "error")


class LLMCallError(Exception):
    """One call failed in a way that retrying will not fix, such as a 400."""


class TransientLLMError(LLMCallError):
    """One call failed in a way that may pass on a retry: a 429, a 5xx, or a timeout.

    retry_after is the wait in seconds that the server asked for, if any.
    """

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class FatalLLMError(Exception):
    """A failure that every later call would repeat, such as a rejected API key.

    It stops the run rather than being recorded against each call.
    """


@dataclass(frozen=True)
class Usage:
    """Tokens one call used, as the provider reported them."""

    input_tokens: int = 0  # input billed at the full rate
    output_tokens: int = 0  # includes any thinking
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass
class Completion:
    """What a provider returned for one request, before the answer is read."""

    text: str
    stop_reason: str  # "end_turn", "max_tokens", "refusal", or the provider's own value
    usage: Usage
    latency_ms: float
    refusal: str | None = None  # the refusal's category and explanation, if given


@dataclass
class LLMResponse:
    """The result of one classification call."""

    present: frozenset | None  # clause types found, or None when the call failed
    outcome: str  # one of OUTCOMES
    error: str | None = None
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    cost_usd: float | None = 0.0  # None when the model is not priced

    @property
    def failed(self) -> bool:
        return self.present is None


class LLMClient:
    """One request per call to classify(), answered in the shared JSON format.

    Subclasses implement send() for one provider.
    """

    provider = ""

    def __init__(self, model: str, max_tokens: int):
        self.model = model
        self.max_tokens = max_tokens
        self.price = price_for(model)
        if self.price is None:
            logger.warning("No listed price for %s, so its calls are reported as not priced", model)

    def send(self, excerpt: str, clause_types: Sequence[str]) -> Completion:
        """Make one request about one excerpt.

        Raises:
            FatalLLMError: when no later call could succeed either
            TransientLLMError: when the same call may succeed if retried
            LLMCallError: when this call failed and retrying will not help
        """
        raise NotImplementedError

    def classify(self, excerpt: str, clause_types: Sequence[str]) -> LLMResponse:
        """Ask which of clause_types appear in one excerpt."""
        start = time.perf_counter()
        try:
            completion = self.send(excerpt, clause_types)
        except LLMCallError as exc:
            return LLMResponse(
                None,
                "error",
                error=str(exc),
                latency_ms=(time.perf_counter() - start) * 1000,
                cost_usd=None if self.price is None else 0.0,
            )
        return self.read(completion, clause_types)

    def read(self, completion: Completion, clause_types: Sequence[str]) -> LLMResponse:
        """Turn a completion into a response, checking why the model stopped first."""
        usage = completion.usage
        common = {
            "text": completion.text,
            "usage": usage,
            "latency_ms": completion.latency_ms,
            "cost_usd": cost_usd(
                self.price,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_read_tokens,
                usage.cache_write_tokens,
            ),
        }
        if completion.stop_reason == "refusal":
            detail = completion.refusal or "no details given"
            return LLMResponse(None, "refusal", error=f"refused: {detail}", **common)
        if completion.stop_reason == "max_tokens":
            error = f"stopped at max_tokens ({self.max_tokens}) before the answer was complete"
            return LLMResponse(None, "truncated", error=error, **common)
        if completion.stop_reason != "end_turn":
            error = f"unexpected stop_reason {completion.stop_reason!r}"
            return LLMResponse(None, "error", error=error, **common)
        try:
            present = parse_answer(completion.text, clause_types)
        except ValueError as exc:
            return LLMResponse(None, "unparseable", error=str(exc), **common)
        return LLMResponse(present, "ok", **common)


def make_client(settings, http_client=None) -> LLMClient:
    """Build the client for settings.provider from an LLMConfig.

    http_client replaces the SDK's HTTP client, which is how tests mock the API.
    """
    if settings.provider == "anthropic":
        from utils.anthropic_client import AnthropicClient

        return AnthropicClient(
            settings.resolved_model,
            api_key=settings.api_key,
            max_tokens=settings.max_tokens,
            effort=settings.effort,
            temperature=settings.temperature,
            timeout=settings.timeout,
            http_client=http_client,
        )
    if settings.provider == "openai":
        from utils.openai_client import OpenAIClient

        return OpenAIClient(
            settings.resolved_model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            timeout=settings.timeout,
            http_client=http_client,
        )
    raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {settings.provider!r}")
