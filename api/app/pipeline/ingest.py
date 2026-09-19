"""Stage 1: PDF -> text, with the offsets that make every later citation provable.

THE CENTRAL INVARIANT
---------------------
This module produces one `raw_text` string for the whole document, and a list of
lines each carrying `char_start` / `char_end` offsets into that string, such that:

    raw_text[line.char_start:line.char_end] == line.text

for every line, always. Clauses inherit those offsets in stage 2.

Why go to this trouble instead of just storing each clause's text on its own?
Because "the model quoted this, and here is the same text sitting at offset
14,203 of the document we actually parsed" is a fundamentally stronger claim
than "the model quoted this, and we also have a copy of some text". The offsets
make the clause store and the source document impossible to silently diverge -
a copy can drift after an edit, a slice cannot. The test in
`tests/test_ingest.py` asserts this for every line of the golden policy.

WHY NO LLM HERE
---------------
Text extraction is a solved, deterministic problem. Asking a model to do it
would introduce transcription errors into the one artefact that everything else
is checked against, and make the pipeline slow and unreproducible for no gain.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# PyMuPDF encodes span styling as a bit field. Bit 4 (value 16) is the bold
# flag. We also check the font name, because some PDFs carry weight in the font
# name without setting the flag.
_FLAG_BOLD = 1 << 4

# Running headers and footers sit in these top and bottom shares of the page.
# Measured in M15: every one on three real policies was within 10% of an edge,
# and body text on the tightest of them began at 13%.
MARGIN_BAND = 0.12
# A margin line repeated on at least this share of pages is page furniture.
# Real headers and footers were on 100% of pages; the most repeated real
# content, list markers like "i.", reached 50% and never sat only in a margin.
FURNITURE_PAGE_SHARE = 0.5

# Digits are masked before comparing lines, so "7 / 25" and "8 / 25" match.
_RE_DIGITS = re.compile(r"\d+")
# Tab and the other C0 control characters. One real policy put "\t \x07" between
# every clause number and its text, and the bell character (\x07) hid the
# capital letter that the segmenter reads as evidence of a clause number.
_RE_CONTROL = re.compile(r"[\x00-\x1f]")


@dataclass
class Line:
    """One visual line of text, with the styling that reveals document structure.

    `size` and `bold` are what let stage 2 tell a heading from body text without
    guessing, which is why extraction keeps them rather than returning bare text.
    """

    text: str
    page: int  # 0-indexed
    size: float  # font size in points, of the line's largest span
    bold: bool
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in PDF points
    char_start: int
    char_end: int
    # Per-span rectangles, kept so the future PDF highlighter can draw exact
    # boxes. Unused by the current side-by-side UI; storing it now avoids a
    # data migration later.
    span_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)


@dataclass
class IngestResult:
    raw_text: str
    lines: list[Line]
    page_count: int
    # Character offset at which each page begins, for page-number lookups.
    page_starts: list[int]

    def body_size(self) -> float:
        """The most common font size = body text.

        Stage 2 needs a baseline to compare against, and it must be derived from
        the document rather than hard-coded: a policy typeset at 8pt and one at
        11pt both have headings, just at different absolute sizes. Measuring the
        mode makes the segmenter scale-independent.

        Weighted by character count, not line count, so a document with many
        short headings does not mistake heading size for body size.
        """
        weights: dict[float, int] = {}
        for line in self.lines:
            key = round(line.size, 1)
            weights[key] = weights.get(key, 0) + len(line.text)
        return max(weights, key=lambda k: weights[k]) if weights else 10.0


def _page_lines(page: fitz.Page, page_no: int) -> list[Line]:
    """One page's lines, in reading order, before offsets are assigned."""
    lines: list[Line] = []
    # PyMuPDF's own block order, deliberately unsorted. It follows the order
    # the typesetter wrote the text, which on all three real policies measured
    # in M15 was reading order: a column at a time, tables row by row. Sorting
    # by position, as this once did, read a two-column page straight across and
    # spliced the columns together.
    blocks = [b for b in page.get_text("dict")["blocks"] if b.get("lines")]

    for block in blocks:
        for raw_line in block["lines"]:
            spans = [s for s in raw_line["spans"] if s["text"].strip()]
            if not spans:
                continue

            text = _RE_CONTROL.sub(" ", "".join(s["text"] for s in spans)).strip()
            if not text:
                continue

            # A line's "size" is its largest span: a line mixing a bold 13pt
            # heading with a stray 9pt character is a heading.
            largest = max(spans, key=lambda s: s["size"])
            bold = any(
                (s["flags"] & _FLAG_BOLD) or "bold" in s["font"].lower()
                for s in spans
            )
            lines.append(
                Line(
                    text=text,
                    page=page_no,
                    size=round(largest["size"], 2),
                    bold=bold,
                    bbox=tuple(round(v, 2) for v in raw_line["bbox"]),
                    char_start=0,  # assigned once furniture is removed
                    char_end=0,
                    span_bboxes=[tuple(round(v, 2) for v in s["bbox"]) for s in spans],
                )
            )
    return lines


def _furniture(pages: list[list[Line]], heights: list[float]) -> set[int]:
    """ids of the lines that are running headers, footers and page numbers.

    They are not policy text, and left in they do two kinds of damage: a header
    beginning with its page number is read as a numbered clause, and one that
    falls inside a clause spanning a page break sits between two halves of a
    sentence, so a quotation of that sentence can no longer be found in it.

    Recognised by repetition in a margin, not by wording: no list of phrases
    could anticipate every insurer's header, but every insurer's header repeats.
    """
    def margin_key(line: Line) -> tuple[str, str] | None:
        height = heights[line.page]
        if line.bbox[1] < height * MARGIN_BAND:
            band = "top"
        elif line.bbox[3] > height * (1 - MARGIN_BAND):
            band = "bottom"
        else:
            return None
        return band, _RE_DIGITS.sub("#", " ".join(line.text.split()))

    pages_with: dict[tuple[str, str], set[int]] = {}
    for lines in pages:
        for line in lines:
            if key := margin_key(line):
                pages_with.setdefault(key, set()).add(line.page)

    # At least 3 pages, so a two-page document never loses a line to this.
    needed = max(3, len(pages) * FURNITURE_PAGE_SHARE)
    repeated = {key for key, seen in pages_with.items() if len(seen) >= needed}
    return {id(line) for lines in pages for line in lines if margin_key(line) in repeated}


def ingest(pdf_path: str | Path) -> IngestResult:
    """Extract text, styling and offsets from a PDF."""
    doc = fitz.open(pdf_path)
    try:
        # Two passes, because furniture is only recognisable by comparing pages.
        pages = [_page_lines(doc[n], n) for n in range(doc.page_count)]
        furniture = _furniture(pages, [doc[n].rect.height for n in range(doc.page_count)])

        chunks: list[str] = []
        lines: list[Line] = []
        page_starts: list[int] = []
        cursor = 0  # running character offset into the joined raw_text

        for page_lines in pages:
            page_starts.append(cursor)
            for line in page_lines:
                if id(line) in furniture:
                    continue
                line.char_start = cursor
                line.char_end = cursor + len(line.text)
                chunks.append(line.text)
                lines.append(line)
                # +1 for the "\n" that joins lines in raw_text. Keeping this in
                # lockstep with the join below is what upholds the slice-back
                # invariant; change one and you must change both.
                cursor += len(line.text) + 1

        return IngestResult(
            raw_text="\n".join(chunks),
            lines=lines,
            page_count=doc.page_count,
            page_starts=page_starts,
        )
    finally:
        doc.close()
