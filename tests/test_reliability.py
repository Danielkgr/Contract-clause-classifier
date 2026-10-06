"""Tests for retries with backoff and the on-disk response cache, with the API mocked."""

import pytest

from config import LLMConfig
from mock_api import MockAPI, error, message
from utils.llm_client import RetryPolicy, make_client
from utils.response_cache import ResponseCache, cache_key

TYPES = ["Governing Law", "Insurance"]


class Sleeps(list):
    """Records requested waits instead of waiting."""

    def __call__(self, seconds):
        self.append(seconds)


def test_a_transient_error_is_retried_after_a_backoff():
    sleeps = Sleeps()
    api = MockAPI(error(529, "overloaded_error"), (200, message()))
    response = api.client(sleep=sleeps).classify("text", TYPES)
    assert response.outcome == "ok"
    assert response.retries == 1
    assert len(api.requests) == 2
    assert len(sleeps) == 1 and sleeps[0] > 0


def test_a_rate_limit_waits_at_least_as_long_as_the_server_asks():
    sleeps = Sleeps()
    status, body = error(429, "rate_limit_error")
    api = MockAPI((status, body, {"retry-after": "7"}), (200, message()))
    response = api.client(sleep=sleeps).classify("text", TYPES)
    assert response.outcome == "ok"
    assert sleeps[0] >= 7


def test_retries_stop_after_the_last_attempt_and_the_call_fails():
    sleeps = Sleeps()
    api = MockAPI(*[error(500, "api_error")] * 3)
    response = api.client(retry=RetryPolicy(attempts=3), sleep=sleeps).classify("text", TYPES)
    assert response.present is None
    assert response.outcome == "error"
    assert "gave up after 3 attempts" in response.error
    assert len(api.requests) == 3 and len(sleeps) == 2


def test_the_sdk_does_not_retry_on_its_own():
    api = MockAPI(error(529, "overloaded_error"))
    api.client(retry=RetryPolicy(attempts=1)).classify("text", TYPES)
    assert len(api.requests) == 1


def test_a_rejected_request_is_not_retried():
    api = MockAPI(error(400, "invalid_request_error"))
    response = api.client(sleep=Sleeps()).classify("text", TYPES)
    assert response.outcome == "error"
    assert len(api.requests) == 1


def test_backoff_doubles_up_to_a_cap_with_jitter():
    policy = RetryPolicy(base_delay=2.0, max_delay=10.0)
    for retry, full in [(1, 2.0), (2, 4.0), (3, 8.0), (4, 10.0), (9, 10.0)]:
        assert full / 2 <= policy.delay(retry) <= full


def test_a_rerun_reads_every_answer_from_the_cache(tmp_path):
    cache = ResponseCache(tmp_path)
    first = MockAPI((200, message())).client(cache=cache).classify("text", TYPES)
    rerun_api = MockAPI()  # any request would fail, as there are no replies queued
    again = rerun_api.client(cache=cache).classify("text", TYPES)
    assert rerun_api.requests == []
    assert again.cached and not first.cached
    assert again.present == first.present
    # The original call's measured latency and cost are kept
    assert again.latency_ms == pytest.approx(first.latency_ms)
    assert again.cost_usd == pytest.approx(first.cost_usd)


def test_errors_are_not_cached_so_a_rerun_retries_them(tmp_path):
    cache = ResponseCache(tmp_path)
    api = MockAPI(error(400, "invalid_request_error"))
    assert api.client(cache=cache).classify("text", TYPES).outcome == "error"
    rerun = MockAPI((200, message()))
    assert rerun.client(cache=cache).classify("text", TYPES).outcome == "ok"
    assert len(rerun.requests) == 1


def test_refusals_are_cached_because_they_are_what_the_model_did(tmp_path):
    cache = ResponseCache(tmp_path)
    MockAPI((200, message(text=None, stop_reason="refusal"))).client(cache=cache).classify(
        "text", TYPES
    )
    rerun = MockAPI()
    assert rerun.client(cache=cache).classify("text", TYPES).outcome == "refusal"
    assert rerun.requests == []


def test_the_cache_key_changes_with_anything_that_shapes_the_request():
    base = dict(provider="anthropic", model="claude-opus-5-5", clause_types=TYPES)
    base.update(settings={"max_tokens": 2048, "effort": "low"}, excerpt="text")
    key = cache_key(**base)
    assert cache_key(**base) == key
    for change in [
        {"provider": "openai"},
        {"model": "claude-haiku-4-5"},
        {"clause_types": ["Insurance"]},
        {"settings": {"max_tokens": 2048, "effort": "high"}},
        {"excerpt": "other text"},
        {"prompt_version": "0"},
    ]:
        assert cache_key(**{**base, **change}) != key, change


def test_an_unreadable_cache_file_is_ignored(tmp_path):
    cache = ResponseCache(tmp_path)
    cache.put("ab" + "0" * 62, {"x": 1})
    (tmp_path / "ab" / ("ab" + "0" * 62 + ".json")).write_text("{not json")
    assert cache.get("ab" + "0" * 62) is None
    assert cache.get("cd" + "0" * 62) is None


def test_make_client_uses_the_configured_cache_unless_told_not_to(tmp_path):
    settings = LLMConfig(api_key="test", cache_dir=str(tmp_path), max_attempts=3)
    client = make_client(settings)
    assert client.cache.directory == tmp_path
    assert client.retry.attempts == 3
    assert make_client(settings, use_cache=False).cache is None
