"""Stage 2: group lines into clauses. Contains no LLM, on purpose.

WHY A MODEL IS THE WRONG TOOL FOR THIS
--------------------------------------
It is tempting to hand the whole document to a model and ask "split this into
clauses". That fails on all four axes that matter here:

  Reproducible - two runs would give different boundaries, so every downstream
                 number (clause counts, scores, eval metrics) becomes noise.
  Verifiable   - a model that paraphrases while splitting silently destroys the
                 character offsets that make citations provable.
  Fast         - a 12,000 character policy would be re-read on every run.
  Debuggable   - when a boundary is wrong you can step through a rule; you
                 cannot step through a forward pass.

Layout already encodes the structure. A policy is typeset with headings in
larger, bolder type and clauses under numbers like "4.2" precisely so a human
can navigate it. Reading that layout is a parsing problem, not a language
problem, so this module reads it directly.

HOW BOUNDARIES ARE FOUND
------------------------
Each line is classified as SECTION heading, CLAUSE heading, or BODY, using:

  1. Font size relative to the document's own body size. Relative, never
     absolute - a policy set in 8pt and one set in 11pt both have headings, and
     `IngestResult.body_size()` measures the baseline per document so the same
     thresholds work for both.
  2. Clause numbering at the start of a line ("4.2", "(a)", "(iv)").
  3. Known IRDAI section names, as a fallback for policies whose headings carry
     no styling at all.
  4. A character cap, so a pathological document still yields workable chunks
     instead of one 40-page clause.
"""

import re
from dataclasses import dataclass, field

from app.pipeline.ingest import IngestResult, Line

# A heading must be this much larger than body text to count as a SECTION.
# 1.20 sits comfortably between the two real ratios in a typical policy
# (section headings ~1.35x body, clause headings ~1.10x), and is validated by
# the golden-policy test rather than chosen by feel.
SECTION_SIZE_RATIO = 1.20

# Clause headings are barely larger than body text, so size alone cannot
# identify them; numbering does the work and size only corroborates.
CLAUSE_SIZE_RATIO = 1.02

# A numbered line shorter than this is treated as a standalone heading
# ("4.2 Room Rent Limit"). Longer, and the number is inline with the clause
# body, which is how many real policies are typeset.
HEADING_MAX_CHARS = 90

# Split anything longer than this. ~3000 chars is roughly 750 tokens: still a
# comfortable single unit for the analyzer, while capping the damage done by a
# document with no detectable structure at all.
MAX_CLAUSE_CHARS = 3000

# "4.", "4.2", "4.2.1" followed by whitespace.
_RE_DECIMAL = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(?=\S)")
# A bare integer with no dot and no trailing period ("24 hours ...") is NOT
# clause numbering - it is prose that happens to start with a number, which
# occurs constantly in insurance text ("24 hours", "10 in-patient beds",
# "15 days"). Distinguishing these is what keeps the numbering signal usable
# on its own.
#
# It was once weak evidence, believed when the line was bold. On real wordings
# that let every bold quantity through: "24 Months waiting period", "1 year
# Tenure", and table headers like "75 Lakhs" all became clauses. No document
# measured - three real, three synthetic - numbers a clause with a bare integer.
_RE_BARE_INT = re.compile(r"^\d+\s")
# "(a)", "(iv)", "(12)"
_RE_PAREN = re.compile(r"^\((?:[a-zA-Z]|[ivxlIVXL]+|\d+)\)\s+(?=\S)")
# A line holding nothing but a marker: "2.", "1.2.", "(a)". Two of three real
# policies set the number a tab stop away from its words, and extraction
# returns it as a line of its own - which no pattern above matches, because
# each needs text after the number.
_RE_MARKER_ONLY = re.compile(r"^(?:\d+(?:\.\d+)*\.?|\((?:[a-zA-Z]|[ivxlIVXL]+|\d+)\))$")
# A number with no level above it, "7." or "(a)", as opposed to "4.2". Only
# these are ambiguous: a list inside a clause is numbered 1, 2, 3 too, but no
# list is numbered 5.1.3.
_RE_SINGLE_LEVEL = re.compile(r"^(?:\d+\.?|\(.+\))$")

# When at least this share of a document's numbered lines are bold, bold is how
# that document marks a clause, and a plain single-level number is a list item.
# Measured shares: 0.00 for the unstyled test policy; 0.24, 0.25 and 0.32 for
# three real wordings, whose clause numbers are bold but whose many plain lines
# that merely start with a number ("24 hours", table rows) dilute the share.
# 0.10 sits in the gap between the two groups, not inside either.
BOLD_NUMBERING_SHARE = 0.10
# "Clause 4.2", "Section 3", "Article II", "LIST I - Items for which coverage is
# not available". The List form is IRDAI's standard annexure of non-payable
# items, used by two of three real wordings measured; without it Star's four
# lists ran together under the benefits table and were cut every 3,000
# characters, and the piece holding the never-paid items was read as a payout
# cap. The closing \b is what keeps "List Items" from reading as "List I".
_RE_LABELLED = re.compile(r"^(?:Clause|Section|Article|Part|List)\s+[\dIVXLivxl]+\b", re.I)

# A structural heading: "SECTION 7", "PART II", "CHAPTER 3", in caps.
# This is the PRIMARY unstyled-section signal, and it is deliberately structural
# rather than vocabulary-based. An earlier version relied only on the word list
# below and missed "SECTION 7 - GENERAL PROVISIONS", because the list happened
# to contain "general condition" but not "provisions". The heading was then
# parsed as a clause, and every clause under it was filed against the previous
# section. A word list can only ever recognise the headings someone remembered
# to add; "SECTION <number>" recognises the shape of a heading.
_RE_SECTION_HEAD = re.compile(
    r"^(?:SECTION|PART|CHAPTER|SCHEDULE|ANNEXURE)\s+[\dIVXL]+\b"
)

# Section names that recur across IRDAI health policies. A secondary signal for
# policies whose headings carry neither styling nor a "SECTION n" prefix.
_SECTION_WORDS = (
    "definition", "scope of cover", "coverage", "benefit", "waiting period",
    "exclusion", "condition", "provision", "claim procedure", "claim",
    "grievance", "annexure", "schedule", "limit", "co-payment", "sub-limit",
    "general", "renewal", "portability", "termination", "premium",
)
# Whole words, optionally plural. Matched as substrings, "limit" found a heading
# in "STAR HEALTH AND ALLIED INSURANCE COMPANY LIMITED" and "condition" one in
# "AIR CONDITIONER CHARGES", a row of a table of non-payable items.
_RE_SECTION_WORD = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in _SECTION_WORDS) + r")s?\b"
)


@dataclass
class Segment:
    """One clause: a contiguous span of the document with a role in the policy."""

    order_idx: int
    section_path: str
    # The clause's own number ("4.2"), when the document provides one. Kept as
    # a first-class field rather than parsed back out of `heading`, because
    # many policies run the number inline with the body and have no separate
    # heading at all - "4.2" is the stable identifier, the heading is optional.
    number: str
    heading: str
    text: str
    page_start: int
    page_end: int
    char_start: int
    char_end: int
    bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


def _numbering(text: str) -> tuple[str, bool] | None:
    """Return (number, is_strong) if the line starts with clause numbering.

    "Strong" means the numbering alone is enough to declare a clause boundary,
    with no help from font styling. That distinction is essential: many real
    policy PDFs are generated with a single uniform font, so a rule that
    *requires* styling finds nothing at all in them.

    Strong:
      "(a)", "(iv)"      - a body line practically never starts this way
      "Clause 4.2"       - explicitly labelled
      "4.2 Room Rent"    - dotted number followed by a capital

    Weak (needs styling to corroborate):
      "1.5 lakhs and ..." - a dotted *quantity*, not a clause number

    Not numbering at all:
      "24 hours of ..."  - a bare integer, however it is styled

    The capital-letter test is what separates "4.2 Room Rent" from "1.5 lakhs".
    Clause numbers are followed by a heading or a new sentence, so the next
    character is upper case; a measurement is followed by its unit in lower case.
    """
    if m := _RE_PAREN.match(text):
        return m.group(0).strip(), True
    if m := _RE_LABELLED.match(text):
        # Whitespace collapsed: Niva sets "List  I" with two spaces, and this
        # string becomes the id the model cites.
        return " ".join(m.group(0).split()), True
    if _RE_BARE_INT.match(text):
        return None
    if m := _RE_DECIMAL.match(text):
        rest = text[m.end() :]
        looks_numbered = bool(rest) and rest[0].isupper()
        return m.group(0).strip(), looks_numbered
    return None


def _looks_like_section(line: Line, body_size: float) -> bool:
    """A section heading: big and bold, or an unmistakable named heading."""
    if line.size >= body_size * SECTION_SIZE_RATIO and line.bold:
        return True
    # Fallbacks for unstyled documents. Length-capped so a shouty sentence in
    # the body cannot masquerade as a heading.
    stripped = line.text.strip()
    if len(stripped) <= 70 and stripped.isupper():
        if _RE_SECTION_HEAD.match(stripped):
            return True
        if _RE_SECTION_WORD.search(stripped.lower()):
            return True
    return False


def _as_read(lines: list[Line], i: int) -> str:
    """A line's text as a reader sees it: a lone marker joined to the words beside it.

    "2." and "Specified disease waiting period" are two lines to the extractor
    and one line to anyone looking at the page. Only a line on the same row and
    to the right is joined; a marker whose next line is lower down is left alone.
    """
    line = lines[i]
    if i + 1 < len(lines) and _RE_MARKER_ONLY.match(line.text):
        mate = lines[i + 1]
        half_height = (line.bbox[3] - line.bbox[1]) / 2
        if (mate.page == line.page
                and abs(mate.bbox[1] - line.bbox[1]) < half_height
                and mate.bbox[0] > line.bbox[0]):
            return f"{line.text} {mate.text}"
    return line.text


def _numbers_are_bold(lines: list[Line], body_size: float) -> bool:
    """Does this document set its clause numbers in bold?

    Decided per document, like body size, because typesetting varies: if a
    policy bolds its clause numbers, a number in plain type is telling you it
    is not one. A policy with no bold at all says nothing either way, and the
    numbering rules apply unchanged.
    """
    numbered = [line for i, line in enumerate(lines)
                if not _looks_like_section(line, body_size) and _numbering(_as_read(lines, i))]
    return bool(numbered) and sum(l.bold for l in numbered) / len(numbered) >= BOLD_NUMBERING_SHARE


def _split_oversized(seg: Segment, raw_text: str) -> list[Segment]:
    """Break a clause that exceeds MAX_CLAUSE_CHARS at newline boundaries.

    Splitting on newlines (rather than mid-sentence) keeps every piece a clean
    slice of raw_text, so the offset invariant survives the split.
    """
    if len(seg.text) <= MAX_CLAUSE_CHARS:
        return [seg]

    pieces: list[Segment] = []
    start = seg.char_start
    while start < seg.char_end:
        end = min(start + MAX_CLAUSE_CHARS, seg.char_end)
        if end < seg.char_end:
            # Prefer the last newline inside the window so we cut between lines.
            nl = raw_text.rfind("\n", start, end)
            if nl > start:
                end = nl
        pieces.append(
            Segment(
                order_idx=seg.order_idx,
                section_path=seg.section_path,
                number=seg.number,
                heading=seg.heading if not pieces else f"{seg.heading} (cont.)",
                text=raw_text[start:end],
                page_start=seg.page_start,
                page_end=seg.page_end,
                char_start=start,
                char_end=end,
                bboxes=seg.bboxes if not pieces else [],
            )
        )
        start = end + 1  # skip the newline we cut on
    return pieces


def segment(result: IngestResult) -> list[Segment]:
    """Split an ingested document into clauses."""
    body_size = result.body_size()

    segments: list[Segment] = []
    section = ""
    # Lines accumulated for the clause currently being built.
    buffer: list[Line] = []
    heading = ""
    number = ""

    def flush() -> None:
        nonlocal buffer, heading, number
        if not buffer:
            return
        start = buffer[0].char_start
        end = buffer[-1].char_end
        seg = Segment(
            order_idx=0,  # assigned after all splitting is done
            section_path=section,
            number=number,
            heading=heading,
            # The slice, not a re-join of the lines. This is what guarantees
            # `raw_text[char_start:char_end] == text` holds for clauses too.
            text=result.raw_text[start:end],
            page_start=buffer[0].page,
            page_end=buffer[-1].page,
            char_start=start,
            char_end=end,
            bboxes=[bbox for line in buffer for bbox in line.span_bboxes],
        )
        segments.extend(_split_oversized(seg, result.raw_text))
        buffer = []
        heading = ""
        number = ""

    lines = result.lines
    bold_numbers = _numbers_are_bold(lines, body_size)

    for i, line in enumerate(lines):
        # Order matters: section headings are checked first, because a section
        # heading like "4. EXCLUSIONS" also matches the clause numbering
        # pattern. Size is what tells them apart.
        if _looks_like_section(line, body_size):
            flush()
            section = line.text.strip()
            continue

        text = _as_read(lines, i)
        numbering = _numbering(text)
        # Strong numbering stands on its own. Weak numbering is only believed
        # when the styling backs it up. Getting this the wrong way round - as
        # this code originally did, requiring styling in every case - silently
        # loses every clause in an unstyled PDF.
        is_clause_start = numbering is not None and (
            numbering[1]
            or line.bold
            or line.size >= body_size * CLAUSE_SIZE_RATIO
        )
        # The opposite mistake, in a document that does use styling: its list
        # items ("12. Hernia of all types", under a waiting-period heading) and
        # a wrapped phone number ("2255. Senior Citizens may call") are all
        # "number, then a capital" - and all in plain type.
        if (is_clause_start and bold_numbers and not line.bold
                and _RE_SINGLE_LEVEL.match(numbering[0])):
            is_clause_start = False

        if is_clause_start:
            flush()
            number = numbering[0].rstrip(".")
            # Short numbered line -> a standalone heading. Long one -> the
            # number is inline with the body, so there is no separate heading.
            if len(text) <= HEADING_MAX_CHARS:
                heading = text.strip()
            buffer = [line]
            continue

        buffer.append(line)

    flush()

    for idx, seg in enumerate(segments):
        seg.order_idx = idx
    return segments
