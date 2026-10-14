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


# A hyphen with whitespace after it, between two word characters: a word the
# PDF broke at the end of a line ("pre-\nexisting", "para-\nsurgical").
_LINE_END_HYPHEN = re.compile(r"(?<=\w)-\s+(?=\w)")


def plain(text: str) -> str:
    """The same words in plain characters, on one line. Case is kept.

    Ligatures are split ("fi" set as one character becomes two), curly quotes
    and dashes become ASCII, and every run of whitespace, line breaks included,
    becomes one space. None of those say anything about what a clause or a
    question means, so no rule that reads text should see them.
    """
    text = unicodedata.normalize("NFKC", text)
    for fancy, ascii_char in _PUNCTUATION_MAP.items():
        text = text.replace(fancy, ascii_char)
    return _WHITESPACE.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Reduce text to what a quotation check should actually compare.

    Collapses whitespace (PDF reflow), unifies typographic punctuation, and
    lowercases. Deliberately preserves every word and their order.
    """
    return plain(text).lower()


def _quotable(text: str) -> str:
    """`normalize`, with a word broken by a hyphen at a line end rejoined.

    A correct quotation of "pre-existing" failed against a clause printed
    "pre-\\nexisting": 12 places in the two real policies measured. Applied to
    both sides of the comparison, so it cannot make differing words match.

    QUOTATIONS ONLY, and that is a measured decision, not an oversight. The
    same join in `verify_exception` lets through a true exception that the
    line break had been hiding (HDFC's Geography clause: "except under Section
    B-\\n2.8 ..."). Printing it changed 8 HDFC questions' requests and one of
    them, dengue at four months, went from right 3/3 to wrong 3/3 on an
    unrelated clause. So the exception check still reads the unjoined text.
    """
    return _LINE_END_HYPHEN.sub("-", normalize(text))


@dataclass
class QuoteCheck:
    clause_id: str
    quote: str
    verified: bool
    reason: str = ""


def _verify_elided(quote: str, source_text: str) -> bool:
    """True if the quote is several exact runs of one clause with material skipped.

    WHY. A question about an inguinal hernia 14 months into a first policy got
    the right verdict off the right clause, and was shown to the reader marked
    *unverified*. The clause, Star's specified-disease waiting period, is
    lettered A to E, and the model quoted the heading, A, D and E - skipping B
    (enhancement of sum insured) and C (overlap with the pre-existing waiting
    period), neither of which bears on the question. Every word it printed was
    the clause's own. What failed was only the demand that a quotation be one
    unbroken run, and telling a reader that an exactly-copied quotation "does
    not appear in the cited clause" is a false alarm - the one thing a grounding
    check cannot afford, because a check that cries wolf gets ignored.

    WHAT KEEPS THIS HONEST. Three constraints, and the reason for each:

      split only at the model's own line breaks - so it can elide where it
          chose to break a line, never mid-sentence, and cannot assemble a new
          sentence out of scattered phrases
      every piece at least MIN_QUOTE_CHARS - so an answer cannot be stitched
          together out of fragments too short to mean anything on their own
      in document order, and non-overlapping - so the clause's own sequence is
          preserved. A condition cannot be lifted above the sentence that
          qualified it, and no passage can be quoted twice to pad the others

    What it deliberately does NOT do is judge whether the skipped material
    mattered; that is a question about meaning, and this check is about
    provenance. The elision stays visible in the quotation the reader is shown.
    """
    pieces = [p for p in (_quotable(line) for line in quote.split("\n")) if p]
    # One piece would have passed the contiguous check already; reaching here
    # with one piece means it genuinely is not in the clause.
    if len(pieces) < 2:
        return False

    haystack = _quotable(source_text)
    cursor = 0
    for piece in pieces:
        if len(piece) < MIN_QUOTE_CHARS:
            return False
        found = haystack.find(piece, cursor)
        if found == -1:
            return False
        cursor = found + len(piece)
    return True


def verify_quote(quote: str, source_text: str) -> tuple[bool, str]:
    """Check that `quote` appears within `source_text` after normalisation."""
    if not quote or not quote.strip():
        return False, "empty quote"

    normalized_quote = _TRAILING_REF.sub("", _quotable(quote)).strip()
    if len(normalized_quote) < MIN_QUOTE_CHARS:
        return False, f"quote too short to verify (<{MIN_QUOTE_CHARS} chars)"

    # The strictest reading first, so an unbroken quotation is still checked
    # exactly as it always was, and the elision path is only ever a fallback.
    if normalized_quote in _quotable(source_text):
        return True, ""
    if _verify_elided(quote, source_text):
        return True, ""
    return False, "quote does not appear in the cited clause"


# How much of a quotation must survive trimming for the trimmed part to stand.
MIN_KEPT_SHARE = 0.6


def verified_prefix(quote: str, source_text: str) -> str | None:
    """The longest leading part of `quote` that the clause really contains, or None.

    WHY. Two quotations failed verification in every run, and both had the same
    shape: the clause's own words, exactly, with something of the model's
    appended - "…has been accepted BY THE COMPANY" (three words from the clause
    before), and a whole clause followed by the pipeline's own "EXCEPTIONS - …"
    annotation. A retry repeated both byte for byte, so asking again does not
    fix this; the true part is already there to keep.

    Same principle as _TRAILING_REF, generalised: what is kept must match
    exactly, so nothing invented can hide in it. And it must be most of the
    quotation - at least MIN_KEPT_SHARE of it, and never shorter than
    MIN_QUOTE_CHARS - so a true opening followed by a longer invention is not
    turned into a verified quotation by throwing the invention away.

    Returns the kept words as the model wrote them (its casing and line breaks),
    so what the reader sees is still a copy of the model's quotation, shortened.
    """
    tokens = quote.split()
    haystack = _quotable(source_text)
    whole = len(_quotable(quote))
    for k in range(len(tokens) - 1, 0, -1):
        kept = " ".join(tokens[:k]).rstrip(" ,;:")
        needle = _quotable(kept)
        if len(needle) < MIN_QUOTE_CHARS or len(needle) < MIN_KEPT_SHARE * whole:
            return None
        if needle in haystack:
            return kept
    return None


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


# --- exceptions extracted at analysis time ---------------------------------
#
# WHY AN EXTRACTED EXCEPTION NEEDS CHECKING AT ALL
# ------------------------------------------------
# Stage 3 asks the model for each clause's carve-outs, and stage 5 prints them
# under the clause as "EXCEPTIONS - this clause does NOT apply when: ...". That
# line is the pipeline speaking, not the model, so a wrong one is a falsehood
# this system asserts. Measured on the synthetic policy, 5 of the 8 clauses
# given an exception were given a wrong one, in two different ways:
#
#   NOT IN THE CLAUSE    the alcohol exclusion was given "necessitated by an
#                        Accident, Burn or Cancer" - the analysis prompt's own
#                        worked example - so a question about a drunken fall
#                        down the stairs was told the exclusion spares accidents
#
#   REAL TEXT, WRONG ROLE  "reversal of sterilisation" is on the infertility
#                          clause's list of things it EXCLUDES; the whole breach-
#                          of-law exclusion was returned as its own exception
#
# A substring check catches the first kind and not the second. The second is
# caught by the grammar of a carve-out: in a policy wording an exception is
# introduced by a word that says so, earlier in the same sentence.

# The same words CLASSIFY_SYSTEM tells the model to look for. A test holds the
# two lists together.
EXCEPTION_MARKERS = (
    "unless", "except", "other than", "save for", "provided that", "shall not apply",
)
_MARKER = re.compile(r"\b(?:" + "|".join(re.escape(m) for m in EXCEPTION_MARKERS) + r")\b")
# A full stop followed by a space. Not a bare full stop, or "3.1" and "1.5
# lakh" would end a sentence. Erring towards more boundaries only ever drops an
# exception, which is the safe direction: the clause text is still in front of
# the reader, only the emphasis is lost.
#
# Not after an abbreviation either: "rs. 10,000", "toll free no. 1800" and
# "(i.e. natural or legally adopted)" are in the real policies, and each would
# cut a sentence in two and hide an exception word before it. "etc." and
# "ltd." are left out on purpose: they often do end a sentence, and treating a
# real end as none errs the unsafe way, letting the previous sentence's
# exception word reach.
_SENTENCE_END = re.compile(
    r"(?<!\brs)(?<!\bno)(?<!\bsr)(?<!\bdr)(?<!\bviz)(?<!\bi\.e)(?<!\be\.g)[.!?]\s"
)


# A bracket that has opened and closed. An exception word inside it governs
# only what is inside: in "miscarriage (unless due to an accident) and lawful
# medical termination of pregnancy", the "unless" does not reach termination.
# Measured twice on its own and reverted both times (it scored worse); kept
# only together with trailing_exceptions below, which restores the true
# exception whose job the false one had been doing by accident.
_CLOSED_BRACKET = re.compile(r"\([^()]*\)")
# A line holding nothing but a list label: "a.", "ii)", "3.".
#
# Matched on normalised text, where line breaks are already spaces, so it makes
# no difference whether the PDF puts each label on a line of its own or runs
# the items together in a paragraph. The first version split on label-only
# LINES and so read the layout, not the clause.
_ITEM_LABEL = re.compile(r"(?:^|\s)\(?(?:[a-z]|[ivx]{1,4}|\d{1,2})[.)](?=\s|$)")
# An exception word, then one to five plain words running to the end of the item.
_TRAILING_EXCEPTION = re.compile(
    r"\b(?:" + "|".join(re.escape(m) for m in EXCEPTION_MARKERS) + r")\s+((?:[a-z-]+ ){0,4}[a-z-]+)[.;]?$"
)


_MAX_ITEMS = 4


def trailing_exceptions(source_text: str) -> list[str]:
    """Exceptions a clause states at the very end of a list item, read by grammar.

    The model is asked for a clause's exceptions and does not always return
    them all: Star's maternity exclusion ends its first item "... except
    ectopic pregnancy" and that was never extracted. Only the plainest shape is
    read here - a few words after the exception word, with no punctuation,
    running to the end of the item - because anything longer is a condition
    that needs reading, not a phrase that can be copied.

    Nothing is read from a clause with many items. An exception is stored for
    the whole clause, so one that ends item (o) of sixteen general exclusions
    ("... unless necessitated by an accident") would be printed over all
    sixteen. HDFC's general exclusions clause has exactly that item.
    """
    items = [i.strip() for i in _ITEM_LABEL.split(normalize(source_text)) if i.strip()]
    # ponytail: exceptions are per clause, so a long list is skipped outright.
    # 4 = a heading and three items; Star's maternity clause is 3, HDFC's
    # general exclusions 17+, nothing between was measured. Storing exceptions
    # per item would remove the need for this.
    if len(items) > _MAX_ITEMS:
        return []
    return [m.group(1) for m in map(_TRAILING_EXCEPTION.search, items) if m]


def verify_exception(span: str, source_text: str) -> bool:
    """True if `span` is a carve-out the clause actually states.

    It must appear in the clause, and an exception word must come before it in
    the same sentence. No minimum length, unlike a quotation: the exception
    word is what stops a short span from matching by accident.
    """
    needle = normalize(span).strip(" .,;:")
    if not needle:
        return False
    haystack = normalize(source_text)
    start = haystack.find(needle)
    while start != -1:
        sentence_so_far = _SENTENCE_END.split(haystack[:start])[-1]
        if _MARKER.search(_CLOSED_BRACKET.sub(" ", sentence_so_far)):
            return True
        start = haystack.find(needle, start + 1)
    return False
