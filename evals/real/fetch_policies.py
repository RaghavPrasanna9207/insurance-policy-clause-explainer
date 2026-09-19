"""Download the real policy wordings listed in sources.json.

WHY A SCRIPT, AND NOT THE PDFs
------------------------------
A policy wording belongs to the insurer that wrote it, so none is committed to
this repository. What is committed is where each one is published and a SHA-256
fingerprint of the exact bytes that were measured.

WHY THE FINGERPRINT MATTERS
---------------------------
Insurers revise wordings and re-upload them to the same address. Without a
check, a later run would quietly measure a different document and its numbers
would be compared against the old ones as if nothing had changed. A mismatch is
therefore an error, not a warning: the file is kept under a `.changed.pdf` name
for inspection, and the run stops.

Usage:
    python evals/real/fetch_policies.py
"""

import hashlib
import json
import sys
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCES = Path(__file__).with_name("sources.json")
# Under samples/, which .gitignore already covers (`*.pdf`).
DEST = REPO_ROOT / "samples" / "real"


def pdf_path(policy_id: str) -> Path:
    return DEST / f"{policy_id}.pdf"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    policies = json.loads(SOURCES.read_text(encoding="utf-8"))["policies"]
    DEST.mkdir(parents=True, exist_ok=True)
    failed = []

    for p in policies:
        target = pdf_path(p["id"])
        if target.exists() and sha256(target.read_bytes()) == p["sha256"]:
            print(f"ok        {p['id']} (already present)")
            continue

        # Some insurer sites refuse requests without a browser-like user agent.
        response = httpx.get(
            p["url"], follow_redirects=True, timeout=60,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        response.raise_for_status()
        digest = sha256(response.content)

        if digest != p["sha256"]:
            changed = target.with_suffix(".changed.pdf")
            changed.write_bytes(response.content)
            print(f"CHANGED   {p['id']}: expected {p['sha256'][:12]}, got {digest[:12]} "
                  f"- kept as {changed.relative_to(REPO_ROOT)}")
            failed.append(p["id"])
            continue

        target.write_bytes(response.content)
        print(f"fetched   {p['id']} ({len(response.content):,} bytes)")

    if failed:
        sys.exit(f"{len(failed)} wording(s) changed since they were measured: {', '.join(failed)}")


if __name__ == "__main__":
    main()
