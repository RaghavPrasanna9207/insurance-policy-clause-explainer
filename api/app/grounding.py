"""Verifying that a quotation really came from the document.

THE TWO MECHANISMS, AND WHY BOTH ARE NEEDED
-------------------------------------------
This project prevents hallucinated citations structurally rather than by asking
the model nicely. There are two layers, and they catch different failures.

1. **Enum-constrained clause ids** (see `pipeline/scenario.py`). The schema's
   `clause_id` field is an enum of exactly the clause ids placed in that prompt.
   Ollama enforces JSON-Schema enums during sampling, so a citation pointing at
   a clause the model was never shown is *unrepresentable* - there is no path
   through the sampler that produces it.

   What this does NOT prevent: citing a real clause that says something else.
   The id is guaranteed to exist; the claim about it is not.

2. **Verbatim span check** (this module). Any text the model quotes must
   actually appear inside the clause it attributed the quote to. This is what
   catches the second failure: a valid id attached to invented wording.

Neither layer alone is enough. The first makes the address real; the second
makes the content real.

WHY NORMALISE BEFORE COMPARING
------------------------------
A strict `in` test would fail constantly for uninteresting reasons. PDF text
extraction reflows lines, so a clause stored from the document contains newlines
where the original had none. Models also silently normalise typographic
punctuation - a policy's curly quotes and en-dashes come back as ASCII.

Neither difference changes meaning, so both are normalised away. What is NOT
normalised is word content or word order: those are the things a fabricated
quote gets wrong, and the check exists to catch exactly that.
"""

import re
import unicodedata
from dataclasses import dataclass

# Characters a model routinely swaps for their ASCII equivalents. Treating
# these as differences would reject correct quotations.
_PUNCTUATION_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
    " ": " ", "…": "...",
}

_WHITESPACE = re.compile(r"\s+")

# A reference marker the model appends to its OWN quote, e.g.
#   "...inception of the first policy with the Company. [3.2 (3.2)]"
#
# Measured: a quotation that was 233 of 245 characters byte-perfect - the whole
# clause, correctly copied - failed verification solely because of a trailing
# tag like this. Reporting that as "quote does not appear in the cited clause"
# is actively misleading: it implies fabrication where the substance was exact,
# and a check that cries wolf is one people learn to ignore.
#
# Stripping is narrow and safe. Only a bracketed group at the very END is
# removed, and everything before it must still match exactly, so no fabricated
# content can hide behind it.
# The inner class excludes only SQUARE brackets, not parentheses: the observed
# tag was "[3.2 (3.2)]", with parens nested inside. A first attempt excluded
# both and therefore matched nothing, which is worth remembering - a regex that
# silently matches nothing looks identical to one that is not needed.
_TRAILING_REF = re.compile(r"[\s.,;:]*[\[(][^\[\]]{0,60}[\])]\s*$")

# A quotation shorter than this proves nothing. "The Company" appears in every
# clause of every policy, so matching it would verify nothing while reporting
# success - a false green light is worse than no check at all.
MIN_QUOTE_CHARS = 25


def normalize(text: str) -> str:
    """Reduce text to what a quotation check should actually compare.

    Collapses whitespace (PDF reflow), unifies typographic punctuation, and
    lowercases. Deliberately preserves every word and their order.
    """
    text = unicodedata.normalize("NFKC", text)
    for fancy, plain in _PUNCTUATION_MAP.items():
        text = text.replace(fancy, plain)
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass
class QuoteCheck:
    clause_id: str
    quote: str
    verified: bool
    reason: str = ""


def verify_quote(quote: str, source_text: str) -> tuple[bool, str]:
    """Check that `quote` appears within `source_text` after normalisation."""
    if not quote or not quote.strip():
        return False, "empty quote"

    normalized_quote = _TRAILING_REF.sub("", normalize(quote)).strip()
    if len(normalized_quote) < MIN_QUOTE_CHARS:
        return False, f"quote too short to verify (<{MIN_QUOTE_CHARS} chars)"

    if normalized_quote in normalize(source_text):
        return True, ""
    return False, "quote does not appear in the cited clause"


def verify_citations(
    citations: list[dict], clause_text_by_id: dict[str, str]
) -> list[QuoteCheck]:
    """Verify every quotation in a set of citations.

    Returns one QuoteCheck per citation rather than a single pass/fail, so the
    UI can mark the specific claim that could not be verified instead of
    discarding a whole answer that was mostly sound.
    """
    checks: list[QuoteCheck] = []
    for citation in citations:
        clause_id = citation.get("clause_id", "")
        quote = citation.get("quote", "")
        source = clause_text_by_id.get(clause_id)

        if source is None:
            # Should be impossible while the id enum is built from these same
            # clauses. Checked anyway: this is the assumption the whole
            # grounding story rests on, so it is worth failing loudly if it
            # ever stops holding.
            checks.append(
                QuoteCheck(clause_id, quote, False, "cited clause is not in this document")
            )
            continue

        verified, reason = verify_quote(quote, source)
        checks.append(QuoteCheck(clause_id, quote, verified, reason))
    return checks
