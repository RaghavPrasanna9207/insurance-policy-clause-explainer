"""M1 gate, part 1: text extraction and the offset invariant.

None of these tests need Ollama. Extraction is deterministic, which is the whole
reason stage 1 contains no LLM.
"""

from pathlib import Path

from app.pipeline.ingest import ingest


def test_extracts_all_pages_and_text(golden_pdf: Path):
    result = ingest(golden_pdf)

    assert result.page_count == 5
    assert len(result.raw_text) > 10_000
    assert "MEDIGUARD" in result.raw_text
    assert len(result.lines) > 100


def test_every_line_slices_back_byte_identical(golden_pdf: Path):
    """THE core invariant of the whole system.

    Every citation this app ever shows a user is ultimately justified by
    `raw_text[start:end]` returning exactly the stored text. If this ever fails,
    the app can still render confident explanations that point at the wrong part
    of the document - which is worse than not working at all, because it looks
    like it works.
    """
    result = ingest(golden_pdf)

    for line in result.lines:
        assert result.raw_text[line.char_start : line.char_end] == line.text, (
            f"offset drift at page {line.page}: {line.text[:60]!r}"
        )


def test_offsets_are_ordered_and_non_overlapping(golden_pdf: Path):
    """Guards against a subtler bug than mismatched text.

    Offsets could each slice back correctly while still being out of order or
    overlapping - which would make "clause A comes before clause B" wrong, and
    with it the position component of the buriedness score.
    """
    result = ingest(golden_pdf)

    for previous, current in zip(result.lines, result.lines[1:]):
        assert previous.char_end <= current.char_start
        assert current.char_start < current.char_end


def test_font_signals_survive_extraction(golden_pdf: Path):
    """Stage 2 finds clause boundaries from styling, so styling must reach it.

    A PDF library that returned only plain text would make the segmenter blind,
    so this asserts the structural signal is actually present.
    """
    result = ingest(golden_pdf)
    body = result.body_size()

    assert 8.0 < body < 12.0, f"implausible body size {body}"

    headings = [line for line in result.lines if line.bold and line.size > body]
    assert len(headings) >= 40, "expected section + clause headings to be detected"
    assert any("SECTION 1" in line.text for line in headings)


def test_body_size_is_the_dominant_size_not_the_largest(golden_pdf: Path):
    """body_size() is a mode, not a max.

    Weighting by character count matters: headings are numerous but short, so
    counting lines instead of characters could elect a heading size as the
    "body" size and invert every threshold in the segmenter.
    """
    result = ingest(golden_pdf)
    sizes = [line.size for line in result.lines]

    assert result.body_size() == 9.5
    assert result.body_size() < max(sizes)


def test_two_columns_are_read_one_column_at_a_time(tmp_path: Path):
    """Regression test for M15's Failure 48: a two-column page read line by line across.

    Ingest used to sort every block on a page by its vertical position. On a
    two-column policy that put each left-column paragraph next to whichever
    right-column paragraph sat level with it, so a maternity exclusion's words
    ended up inside a different clause. The offset invariant held throughout,
    because it only compares clauses with raw_text - and raw_text was the thing
    assembled in the wrong order.

    So this checks against something ingest did not produce: the order the
    paragraphs were written in. Paragraphs in each column are staggered in
    height, which is what made the sort interleave them on the real document.
    """
    import fitz

    left = [(90, 150, "LEFT-ONE The first paragraph of the left column."),
            (160, 260, "LEFT-TWO The second paragraph of the left column.")]
    right = [(90, 120, "RIGHT-ONE The first paragraph of the right column."),
             (130, 230, "RIGHT-TWO The second paragraph of the right column.")]
    doc = fitz.open()
    page = doc.new_page()
    # Written column by column - the order a reader follows, and the order a
    # typesetting program emits.
    for y0, y1, text in left:
        page.insert_textbox(fitz.Rect(40, y0, 280, y1), text, fontsize=9)
    for y0, y1, text in right:
        page.insert_textbox(fitz.Rect(310, y0, 550, y1), text, fontsize=9)
    pdf = tmp_path / "two-columns.pdf"
    doc.save(pdf)
    doc.close()

    raw = ingest(pdf).raw_text
    positions = [raw.index(label) for label in ("LEFT-ONE", "LEFT-TWO", "RIGHT-ONE", "RIGHT-TWO")]
    assert positions == sorted(positions), f"reading order scrambled: {raw!r}"


def _paged_pdf(path: Path, pages: list[str], header: str = "ACME HEALTH | POLICY WORDINGS") -> Path:
    """A document whose every page carries a running header and a numbered footer."""
    import fitz

    doc = fitz.open()
    for n, body in enumerate(pages, start=1):
        page = doc.new_page()  # 595 x 842 points
        page.insert_text((60, 40), header, fontsize=8)
        page.insert_textbox(fitz.Rect(60, 120, 530, 760), body, fontsize=9)
        page.insert_text((280, 820), f"{n} / {len(pages)}", fontsize=8)
    doc.save(path)
    doc.close()
    return path


def test_running_headers_and_page_numbers_are_not_text(tmp_path: Path):
    """Regression test for M15's Failure 47: page furniture became clauses.

    A real policy's footer "7 / 25" was read as clause number 7 on every page,
    and a header between two halves of a sentence that crossed a page break
    meant a quotation of that sentence could never be found in the clause.
    The reference here is the sentence as written, not anything ingest produced.
    """
    sentence_start = "4.2 Room rent is payable up to one percent of the"
    sentence_end = "sum insured per day for each day of admission."
    pdf = _paged_pdf(tmp_path / "paged.pdf", [
        "4.1 Cover. The Company will pay for hospitalisation.",
        "Note: read the schedule.\n" + sentence_start,
        sentence_end + "\nNote: read the schedule.",
        "4.3 Claims must be notified within 24 hours.",
    ])

    raw = ingest(pdf).raw_text

    assert "ACME HEALTH" not in raw
    assert "/ 4" not in raw
    assert f"{sentence_start} {sentence_end}" in " ".join(raw.split())
    # Repeated body text is not furniture: it is not in a margin.
    assert raw.count("Note: read the schedule.") == 2


def test_a_short_document_keeps_its_header(tmp_path: Path):
    """Furniture is recognised by repetition, so two pages are too few to tell.

    A header on both pages of a two-page document is as likely to be a title
    as a running header, and dropping real text is the worse mistake.
    """
    pdf = _paged_pdf(tmp_path / "short.pdf", ["4.1 Cover.", "4.2 Claims."])
    assert "ACME HEALTH" in ingest(pdf).raw_text


def test_control_characters_do_not_reach_the_text(tmp_path: Path):
    """One real policy put a tab and a bell character (\\x07) after every clause number.

    "13.\\t \\x07Treatments" hid the capital letter the segmenter reads as
    evidence that "13." is a clause number rather than a quantity.
    """
    import fitz

    doc = fitz.open()
    doc.new_page().insert_text((60, 120), "13.\t \x07Treatments in health hydros", fontsize=9)
    pdf = tmp_path / "control.pdf"
    doc.save(pdf)
    doc.close()

    text = ingest(pdf).lines[0].text
    assert not any(ord(c) < 32 for c in text), repr(text)
    assert text.split() == ["13.", "Treatments", "in", "health", "hydros"]


def test_page_starts_align_with_pages(golden_pdf: Path):
    result = ingest(golden_pdf)

    assert len(result.page_starts) == result.page_count
    assert result.page_starts[0] == 0
    assert result.page_starts == sorted(result.page_starts)

    # Each page's first line should begin at or after that page's start offset.
    for page_no, start in enumerate(result.page_starts):
        first = next((l for l in result.lines if l.page == page_no), None)
        if first is not None:
            assert first.char_start >= start
