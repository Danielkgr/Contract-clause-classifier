"""
The provider-neutral part of the zero-shot LLM arm.

A provider module turns one request into a Completion: the answer text, why
the model stopped, the token usage, and the latency.  LLMClient.classify()
reads that into an LLMResponse, which says which clause types the model
found, or why the call failed.  A failed call is never read as "absent".

classify() also retries transient errors with backoff and keeps completed
responses in an on-disk cache, so a re-run is free and a crash loses
nothing.  Claude is reached only through utils/anthropic_client.py, and
OpenAI only through utils/openai_client.py.  make_client() picks one by
provider.
"""

import dataclasses
import logging
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from utils.pricing import cost_usd, price_for
from utils.prompts import parse_answer
from utils.response_cache import ResponseCache, cache_key

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

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, record: dict) -> "Completion":
        return cls(**{**record, "usage": Usage(**record["usage"])})


@dataclass
class LLMResponse:
    """The result of one classification call."""

    present: frozenset | None  # clause types found, or None when the call failed
    outcome: str  # one of OUTCOMES
    error: str | None = None
    text: str = ""
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0  # one attempt, without any wait between retries
    cost_usd: float | None = 0.0  # None when the model is not priced
    cached: bool = False  # read from the response cache, with no API call
    retries: int = 0

    @property
    def failed(self) -> bool:
        return self.present is None


@dataclass(frozen=True)
class RetryPolicy:
    """Exponential backoff with jitter for transient errors."""

    attempts: int = 6  # tries in all, so up to five retries
    base_delay: float = 2.0  # seconds before the first retry, doubling after each
    max_delay: float = 60.0

    def delay(self, retry: int, retry_after: float | None = None) -> float:
        """Seconds to wait before retry number retry, counting from 1."""
        backoff = min(self.max_delay, self.base_delay * 2 ** (retry - 1))
        # Jitter spreads out concurrent workers that failed together
        wait = backoff * (0.5 + random.random() / 2)
        if retry_after is not None:
            wait = max(wait, min(retry_after, self.max_delay))
        return wait


class LLMClient:
    """One request per call to classify(), answered in the shared JSON format.

    Subclasses implement send() and cache_settings() for one provider.
    """

    provider = ""

    def __init__(
        self,
        model: str,
        max_tokens: int,
        cache: ResponseCache | None = None,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.cache = cache
        self.retry = retry or RetryPolicy()
        self.sleep = sleep
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

    def cache_settings(self) -> dict:
        """The request settings that change the answer, for the cache key."""
        return {"max_tokens": self.max_tokens}

    def classify(self, excerpt: str, clause_types: Sequence[str]) -> LLMResponse:
        """Ask which of clause_types appear in one excerpt."""
        key = None
        if self.cache is not None:
            key = cache_key(self.provider, self.model, clause_types, self.cache_settings(), excerpt)
            record = self.cache.get(key)
            if record is not None:
                response = self.read(Completion.from_dict(record), clause_types)
                response.cached = True
                return response

        retries = 0
        while True:
            start = time.perf_counter()
            try:
                completion = self.send(excerpt, clause_types)
                break
            except TransientLLMError as exc:
                if retries + 1 >= self.retry.attempts:
                    error = f"{exc} (gave up after {retries + 1} attempts)"
                    return self._failed(error, start, retries)
                retries += 1
                wait = self.retry.delay(retries, exc.retry_after)
                logger.info("Retry %d for %s in %.1f s: %s", retries, self.model, wait, exc)
                self.sleep(wait)
            except LLMCallError as exc:
                return self._failed(str(exc), start, retries)

        if self.cache is not None:
            self.cache.put(key, completion.to_dict())
        response = self.read(completion, clause_types)
        response.retries = retries
        return response

    def _failed(self, error: str, start: float, retries: int) -> LLMResponse:
        return LLMResponse(
            None,
            "error",
            error=error,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=None if self.price is None else 0.0,
            retries=retries,
        )

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


def make_client(settings, http_client=None, use_cache: bool = True) -> LLMClient:
    """Build the client for settings.provider from an LLMConfig.

    http_client replaces the SDK's HTTP client, which is how tests mock the
    API.  With use_cache false, responses are neither read from nor written
    to the cache.
    """
    common = {
        "api_key": settings.api_key,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "timeout": settings.timeout,
        "http_client": http_client,
        "cache": ResponseCache(settings.cache_dir) if use_cache else None,
        "retry": RetryPolicy(attempts=settings.max_attempts),
    }
    if settings.provider == "anthropic":
        from utils.anthropic_client import AnthropicClient

        return AnthropicClient(settings.resolved_model, effort=settings.effort, **common)
    if settings.provider == "openai":
        from utils.openai_client import OpenAIClient

        return OpenAIClient(settings.resolved_model, base_url=settings.base_url, **common)
    raise ValueError(f"LLM_PROVIDER must be one of {PROVIDERS}, got {settings.provider!r}")
