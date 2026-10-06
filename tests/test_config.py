"""Tests for configuration loading and for how secrets are shown."""

import os

import pytest

from config import LLMConfig, TrainingConfig, load_config, mask_secret


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


@pytest.fixture
def clean_environ(monkeypatch):
    """An environment with no settings in it, restored after the test.

    load_dotenv writes into os.environ, so the test gets its own copy.
    """
    monkeypatch.setattr(os, "environ", {"PATH": os.environ.get("PATH", "")})
    return os.environ


def test_dotenv_file_supplies_settings(tmp_path, clean_environ):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=model-from-dotenv\nNUM_EPOCHS=7\nLLM_API_KEY=sk-test-key\n")
    config = load_config(env_file)
    assert config.llm.model == "model-from-dotenv"
    assert config.training.num_epochs == 7
    assert config.llm.api_key == "sk-test-key"


def test_environment_wins_over_dotenv(tmp_path, clean_environ):
    env_file = tmp_path / ".env"
    env_file.write_text("LLM_MODEL=model-from-dotenv\n")
    clean_environ["LLM_MODEL"] = "model-from-shell"
    assert load_config(env_file).llm.model == "model-from-shell"


def test_a_missing_dotenv_file_leaves_the_defaults(tmp_path, clean_environ):
    config = load_config(tmp_path / "absent.env")
    assert config.llm.model == LLMConfig.model
    assert config.training.batch_size == TrainingConfig.batch_size
    assert config.training.cost_per_hour is None


def test_settings_are_read_when_built_not_at_import():
    assert TrainingConfig.from_env({"BATCH_SIZE": "32"}).batch_size == 32
    assert TrainingConfig.from_env({"FT_COST_PER_HOUR": "0.6"}).cost_per_hour == 0.6
    assert TrainingConfig.from_env({"FT_COST_PER_HOUR": ""}).cost_per_hour is None


def test_a_malformed_number_names_the_variable():
    with pytest.raises(ValueError, match="BATCH_SIZE must be a whole number"):
        TrainingConfig.from_env({"BATCH_SIZE": "eight"})


def test_the_api_key_never_appears_in_the_config_repr():
    llm = LLMConfig.from_env({"LLM_API_KEY": "sk-ant-secret-value-1234"})
    assert "sk-ant-secret" not in repr(llm)
