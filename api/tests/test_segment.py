"""M1 gate, part 2: clause boundaries.

Checked against the golden policy's label file, which was emitted by the same
script that generated the PDF - so the expectations here cannot drift from the
document they describe.
"""

from pathlib import Path

from app.pipeline.ingest import ingest
from app.pipeline.segment import MAX_CLAUSE_CHARS, Segment, segment


def _segments(pdf: Path) -> tuple[list[Segment], str]:
    result = ingest(pdf)
    return segment(result), result.raw_text


def test_finds_every_numbered_clause(golden_pdf: Path, golden_labels: list[dict]):
    """Recall is what matters most here.

    A missed clause is invisible downstream: it is never classified, never
    scored, and never shown. If the missed one happens to be an exclusion, the
    app quietly tells the user their policy is safer than it is.
    """
    segments, _ = _segments(golden_pdf)
    numbers = {s.number for s in segments if s.number}

    missing = [label["number"] for label in golden_labels if label["number"] not in numbers]
    assert not missing, f"segmenter lost clauses: {missing}"


def test_does_not_over_split(golden_pdf: Path, golden_labels: list[dict]):
    """Precision, the mirror of the test above.

    39 real clauses plus one front-matter block (title, UIN, disclaimer) = 40.
    Allowing a small margin keeps the test from being brittle, while still
    failing loudly if the segmenter starts shattering clauses into fragments.
    """
    segments, _ = _segments(golden_pdf)
    assert len(golden_labels) <= len(segments) <= len(golden_labels) + 3


def test_every_clause_slices_back_byte_identical(golden_pdf: Path):
    """The offset invariant, carried through segmentation.

    Stage 1 guarantees it per line. Stage 2 builds clauses by slicing rather
    than by re-joining line text, so the guarantee must survive - this is what
    proves it did.
    """
    segments, raw_text = _segments(golden_pdf)

    for seg in segments:
        assert raw_text[seg.char_start : seg.char_end] == seg.text, (
            f"offset drift in clause {seg.order_idx}: {seg.heading!r}"
        )


def test_clause_text_matches_the_authored_source(golden_pdf: Path, golden_labels: list[dict]):
    """The strongest available check: extracted text vs. what was authored.

    PDF extraction reflows lines, so the comparison normalises whitespace. Any
    difference beyond that means characters were dropped, duplicated, or
    reordered somewhere in stages 1-2.
    """
    segments, _ = _segments(golden_pdf)
    by_number = {s.number: s for s in segments if s.number}

    for label in golden_labels:
        seg = by_number[label["number"]]
        extracted = " ".join(seg.text.split())
        authored = " ".join(label["text"].split())
        assert authored in extracted, f"clause {label['number']} text was altered"


def test_section_paths_are_assigned(golden_pdf: Path, golden_labels: list[dict]):
    """Section context feeds the analyzer's prompt and the UI's breadcrumbs."""
    segments, _ = _segments(golden_pdf)
    by_number = {s.number: s for s in segments if s.number}

    for label in golden_labels:
        assert by_number[label["number"]].section_path == label["section"]


def test_page_numbers_are_correct(golden_pdf: Path, golden_labels: list[dict]):
    """A citation the user cannot find on the stated page is not a citation."""
    segments, _ = _segments(golden_pdf)
    by_number = {s.number: s for s in segments if s.number}

    for label in golden_labels:
        seg = by_number[label["number"]]
        # Labels are 1-indexed for humans; Segment.page_start is 0-indexed.
        assert seg.page_start + 1 <= label["page"] <= seg.page_end + 1


def test_no_clause_exceeds_the_size_cap(golden_pdf: Path):
    """The oversize split is a safety net for documents with no structure.

    It should not fire on a well-formed policy, but nothing may exceed the cap
    regardless, or a single clause could blow the analyzer's context budget.
    """
    segments, _ = _segments(golden_pdf)
    assert all(len(s.text) <= MAX_CLAUSE_CHARS for s in segments)


def test_order_index_follows_document_order(golden_pdf: Path):
    segments, _ = _segments(golden_pdf)

    assert [s.order_idx for s in segments] == list(range(len(segments)))
    for previous, current in zip(segments, segments[1:]):
        assert previous.char_start < current.char_start


# ---------------------------------------------------------------------------
# The unstyled document. These are regression tests for a real failure:
# the segmenter originally REQUIRED font styling to confirm a clause boundary,
# which meant a PDF with one uniform font produced 8 segments instead of 39 -
# every clause silently swallowed into its section. It passed every test above
# while doing so, because those tests only ever ran on the styled PDF.
# ---------------------------------------------------------------------------


def test_finds_every_clause_without_any_font_signal(hostile_pdf, golden_labels):
    """Numbering alone must be enough to find a clause boundary."""
    segments, _ = _segments(hostile_pdf)
    numbers = {s.number for s in segments if s.number}

    missing = [label["number"] for label in golden_labels if label["number"] not in numbers]
    assert not missing, f"unstyled policy lost clauses: {missing}"


def test_hostile_document_really_has_no_styling(hostile_pdf):
    """Guards the guard.

    If the generator ever started emitting bold or varied sizes, the test above
    would keep passing for the wrong reason and the regression could return
    unnoticed.
    """
    from app.pipeline.ingest import ingest

    result = ingest(hostile_pdf)
    assert not any(line.bold for line in result.lines), "hostile PDF has bold text"
    assert len({line.size for line in result.lines}) == 1, "hostile PDF has varied sizes"


def test_sections_found_from_capitalisation_alone(hostile_pdf):
    """The ALL-CAPS + known-section-word fallback, with no size signal to help."""
    segments, _ = _segments(hostile_pdf)
    sections = {s.section_path for s in segments if s.section_path}

    assert len(sections) >= 7
    assert any("EXCLUSION" in s for s in sections)
    assert any("WAITING PERIOD" in s for s in sections)


def test_offsets_hold_on_the_unstyled_document(hostile_pdf):
    segments, raw_text = _segments(hostile_pdf)
    for seg in segments:
        assert raw_text[seg.char_start : seg.char_end] == seg.text


def test_prose_starting_with_a_number_is_not_a_clause():
    """The precision half of the numbering rule, unit-tested directly.

    Insurance text is full of lines that begin with digits - "24 hours",
    "15 days", "1.5 lakhs". Treating those as clause boundaries would shatter
    clauses into fragments. Only a dotted number followed by a capital, or an
    explicit "(a)" / "Clause 4.2" form, counts as strong evidence.
    """
    from app.pipeline.segment import _numbering

    # Strong: real clause numbering, believed with no styling support.
    assert _numbering("4.2 Room Rent Limit")[1] is True
    assert _numbering("1.1 Hospital. Hospital means any institution")[1] is True
    assert _numbering("(a) the Insured Person shall")[1] is True
    assert _numbering("Clause 4.2 shall apply")[1] is True

    # Weak: prose that merely opens with a number.
    assert _numbering("24 hours of hospitalisation is required")[1] is False
    assert _numbering("1.5 lakhs and fifteen in-patient beds")[1] is False
    assert _numbering("15 days from the date of discharge")[1] is False

    # Not numbering at all.
    assert _numbering("The Company shall indemnify") is None


def test_section_headings_recognised_structurally_not_by_vocabulary():
    """Headings are found by shape ("SECTION <n>"), not by a list of known words.

    A word list only recognises headings someone remembered to add. This rule
    was introduced after "SECTION 7 - GENERAL PROVISIONS" slipped through a
    list containing "general condition" but not "provisions", and was then
    parsed as a clause - silently filing every clause beneath it under the
    previous section.
    """
    from app.pipeline.segment import _RE_SECTION_HEAD

    assert _RE_SECTION_HEAD.match("SECTION 7 - GENERAL PROVISIONS")
    assert _RE_SECTION_HEAD.match("PART II - CONDITIONS")
    assert _RE_SECTION_HEAD.match("ANNEXURE III")
    # The word boundary is what stops "SECTIONAL" being read as "SECTION" + "AL".
    assert not _RE_SECTION_HEAD.match("SECTIONAL TITLE HERE")


def test_no_regex_contains_a_control_character():
    """Regression guard for a genuinely invisible bug.

    A pattern once picked up a literal 0x08 byte, because an intended `\b`
    (word boundary) was written through a shell layer that ate one level of
    escaping, leaving Python to parse `\b` as the backspace escape. The pattern
    printed and compiled normally, and silently matched nothing. Control
    characters have no legitimate place in these patterns, so scanning for them
    catches the whole class of mistake.
    """
    import app.pipeline.segment as seg

    for name in dir(seg):
        if not name.startswith("_RE_"):
            continue
        pattern = getattr(seg, name).pattern
        bad = [hex(ord(c)) for c in pattern if ord(c) < 32]
        assert not bad, f"{name} contains control characters {bad}: {pattern!r}"
