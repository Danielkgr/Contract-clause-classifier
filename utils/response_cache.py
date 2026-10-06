"""
Completed API responses kept on disk, so an interrupted run loses nothing.

Each response is one JSON file named by a key over everything that shapes
the request: the provider, the model, the prompt version and text, the
clause types asked about, the request settings, and a hash of the excerpt.
A re-run with the same settings reads every answer from here and makes no
API call.  Only completed responses are kept, refusals and cut-off answers
included, because they are what the model did.  Errors are not kept, so a
re-run retries them.
"""

import hashlib
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

from utils.prompts import PROMPT_VERSION, system_prompt

logger = logging.getLogger(__name__)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_key(
    provider: str,
    model: str,
    clause_types: Sequence[str],
    settings: dict,
    excerpt: str,
    prompt_version: str = PROMPT_VERSION,
) -> str:
    """A key that changes whenever anything that shapes the request changes."""
    material = {
        "provider": provider,
        "model": model,
        "prompt_version": prompt_version,
        # Guards against a wording change made without bumping the version
        "system_sha256": _sha256(system_prompt(clause_types)),
        "clause_types": list(clause_types),
        "settings": settings,
        "excerpt_sha256": _sha256(excerpt),
    }
    return _sha256(json.dumps(material, sort_keys=True))


class ResponseCache:
    """One JSON file per completed response, written atomically."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        """The stored record for key, or None when there is none or it is unreadable."""
        path = self._path(key)
        try:
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable cached response %s: %s", path, exc)
            return None

    def put(self, key: str, record: dict) -> None:
        """Store record under key.  A crash mid-write never leaves a partial file."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(record, fh)
            os.replace(tmp, path)
        except BaseException:
            os.unlink(tmp)
            raise
