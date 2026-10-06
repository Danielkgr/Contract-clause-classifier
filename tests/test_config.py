"""Tests for configuration handling, starting with how secrets are shown."""

from config import mask_secret


def test_mask_secret_shows_at_most_the_last_four_characters():
    key = "sk-ant-api03-" + "x" * 80 + "WXYZ"
    masked = mask_secret(key)
    assert masked == "...WXYZ"
    assert key[:-4] not in masked


def test_mask_secret_hides_short_values_entirely():
    assert mask_secret("short-key-123") == "****"


def test_mask_secret_reports_a_missing_value():
    assert mask_secret(None) == "(not set)"
    assert mask_secret("") == "(not set)"
