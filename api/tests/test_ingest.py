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
