"""Content-addressed cache for LLM responses.

Analysing a 200-clause policy takes 2-5 minutes. Without a cache, every prompt
tweak means re-running all 200 clauses to see the effect on the handful you
actually changed. With one, only genuinely-new work costs time.

The cache is keyed by a hash of *everything the model actually receives*: the
model name, the messages, the JSON schema and the decoding options. That makes
stale reuse impossible by construction - if any input differs, the key differs,
so there is no manual "remember to clear the cache" step to forget.

PROMPT_VERSION IS DELIBERATELY NOT PART OF THE KEY, though it used to be. It was
redundant for safety: a reworded prompt is different message text, so it
already gets a different key. What it did add was harm. Bumping the version on
any prompt change discarded every cached answer, including the ones whose
prompts had not changed at all - so every change regenerated all 40 eval cases,
and a model that does not answer identically twice turned each of them into a
fresh chance to wobble. A change touching three cases could not be told apart
from noise across forty.

Keyed on content alone, an unchanged prompt replays its stored answer and only
the cases a change actually reached are regenerated. The version survives as a
label in reports and the eval history, which is what it is good for.

It lives in its own SQLite file rather than the app database so that
`rm data/llm_cache.db` resets model outputs without touching uploaded documents.
Plain sqlite3 (not SQLModel) keeps it usable from tests and scripts that never
start the app.
"""

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

_CACHE_PATH = Path("data/llm_cache.db")
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the analyze stage runs batches concurrently,
        # so the connection is touched from more than one thread. Every access
        # is serialised by _lock below, which keeps that safe.
        _conn = sqlite3.connect(_CACHE_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS llm_cache ("
            "  key TEXT PRIMARY KEY,"
            "  model TEXT NOT NULL,"
            "  response TEXT NOT NULL,"
            "  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP"
            ")"
        )
        _conn.commit()
    return _conn


def make_key(
    model: str,
    messages: list[dict[str, str]],
    schema: dict[str, Any] | None,
    options: dict[str, Any] | None = None,
) -> str:
    """Hash every input that could change the response.

    `options` is included because decoding parameters change the OUTPUT, not
    merely the runtime. `num_ctx` truncates the prompt when it is too small and
    `num_predict` cuts the response short - both produce a genuinely different
    answer, so a cached result from one setting must never be served for
    another. The same rule the messages enforce for wording, applied to the
    parameters rather than the words.

    sort_keys=True matters: Python preserves dict insertion order, so two
    logically identical schemas built in a different field order would otherwise
    hash differently and silently miss the cache.
    """
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "schema": schema,
            "options": options or {},
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(key: str) -> dict[str, Any] | None:
    with _lock:
        row = _connect().execute(
            "SELECT response FROM llm_cache WHERE key = ?", (key,)
        ).fetchone()
    return json.loads(row[0]) if row else None


def put(key: str, model: str, response: dict[str, Any]) -> None:
    with _lock:
        conn = _connect()
        # INSERT OR REPLACE: a concurrent duplicate request is harmless, since
        # identical keys mean identical content.
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (key, model, response) VALUES (?, ?, ?)",
            (key, model, json.dumps(response)),
        )
        conn.commit()


def stats() -> dict[str, int]:
    with _lock:
        (count,) = _connect().execute("SELECT COUNT(*) FROM llm_cache").fetchone()
    return {"entries": count}


def clear() -> None:
    with _lock:
        conn = _connect()
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
