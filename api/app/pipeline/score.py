"""Stage 4: turn analyses into a ranked list. Pure arithmetic, no LLM.

WHY THIS STAGE EXISTS SEPARATELY
--------------------------------
The problem this project addresses is that policies *obscure* the clauses that
decide claims. "Obscure" is not a synonym for "bad" - it is a claim about how
hard something is to find and understand. That is measurable, and measuring it
is what makes this tool different from a summariser.

So the score has two halves:

  From the model (stage 3)   likelihood, severity  - judgments about meaning
  Computed here              buriedness            - facts about the document

A 7B model can reasonably judge that a 20% senior-citizen co-payment is
financially significant. It cannot reliably tell you that a clause sits 78% of
the way through a document, references three other clauses, and reads at a
university grade level. Those are arithmetic, and arithmetic done by a language
model is arithmetic you cannot trust or reproduce.

THE FORMULA
-----------
    impact = 100 * type_weight * blend * (1 + BURIEDNESS_BOOST * buriedness)
                                       / (1 + BURIEDNESS_BOOST)

    blend      = 0.5 * norm(likelihood) + 0.5 * norm(severity)
    norm(x)    = (x - 1) / 4                     maps 1..5 onto 0..1
    buriedness = weighted blend of four computed signals, 0..1

Dividing by (1 + BURIEDNESS_BOOST) keeps the result inside 0..100 without
clamping, so no two different clauses collapse onto a saturated 100.

Buriedness is a MULTIPLIER, never an additive term. A clearly written,
prominently placed exclusion is still an exclusion and still matters; being
buried makes a consequential clause worse, but cannot by itself make a trivial
clause important. Multiplying preserves that; adding would not.
"""

import re
from dataclasses import dataclass

from app.pipeline.analyze import ClauseAnalysis
from app.pipeline.segment import Segment
from app.taxonomy import ClauseType

# How much a clause's role matters to a claim outcome, independent of content.
#
# Exclusions and conditions sit at the top: either can result in nothing being
# paid at all.
#
# SUB_LIMIT was originally 0.70 and is now 0.85, on an expected-loss argument
# rather than a worst-case one. Conditions and exclusions are CONTINGENT - a
# notice deadline costs you everything, but only if you miss it. A sub-limit is
# CERTAIN: a room-rent cap applies to every claim, automatically, forever, and
# with proportionate deduction it reduces the whole bill rather than just the
# room. Weighing only the worst case systematically understated the clauses
# that, in Indian health insurance, cause the most actual shortfalls.
#
# It stays below CONDITION because a total repudiation is still the worse tail,
# and tail risk is what people are least able to absorb.
#
# Definitions sit at the bottom for DIRECT impact - though a narrow "Hospital"
# definition can gut a coverage clause, which is indirect harm this weighting
# deliberately does not try to capture.
TYPE_WEIGHT = {
    ClauseType.EXCLUSION: 0.95,
    ClauseType.CONDITION: 0.95,
    ClauseType.SUB_LIMIT: 0.85,
    ClauseType.WAITING_PERIOD: 0.80,
    ClauseType.PROCEDURAL: 0.35,
    ClauseType.COVERAGE: 0.30,
    ClauseType.DEFINITION: 0.20,
}

# At most a 25% uplift for a maximally buried clause. Kept modest on purpose:
# burial should reorder clauses of similar consequence, not let an obscure
# definition outrank a plainly stated exclusion.
BURIEDNESS_BOOST = 0.25

# Relative contribution of each buriedness signal.
W_POSITION = 0.20
W_READING = 0.35
W_CROSSREF = 0.25
W_JARGON = 0.20

# Phrases that send a reader somewhere else to find out what a clause means.
# Each one is a hop, and every hop is a chance to give up.
_RE_CROSSREF = re.compile(
    r"\b(?:as defined in|subject to|specified in|listed in|referred to in|"
    r"in accordance with|under (?:clause|section)|pursuant to|"
    r"annexure|schedule|clause \d|section \d)\b",
    re.IGNORECASE,
)

_RE_SENTENCE = re.compile(r"[.!?]+")
_RE_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_RE_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def _syllables(word: str) -> int:
    """Approximate syllable count by counting vowel groups.

    A heuristic, not a dictionary. It is wrong on individual words, but
    Flesch-Kincaid is applied across a whole clause where the errors average
    out, and the score is used only to rank clauses against each other - a
    consistent small bias changes no ordering.
    """
    word = word.lower()
    count = len(_RE_VOWEL_GROUP.findall(word))
    # A trailing silent "e" ("scale") was counted as its own group.
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)


def flesch_kincaid_grade(text: str) -> float:
    """US school grade needed to read this text on one pass.

    0.39*(words/sentence) + 11.8*(syllables/word) - 15.59

    Legal drafting scores high on both terms at once: long sentences stacked
    with subordinate clauses, built from Latinate words ("indemnify",
    "repudiate", "consequential"). That is precisely the obscuring this project
    is trying to surface, which is why difficulty is the heaviest buriedness
    signal.
    """
    words = _RE_WORD.findall(text)
    if not words:
        return 0.0
    sentences = max(len([s for s in _RE_SENTENCE.split(text) if s.strip()]), 1)
    syllables = sum(_syllables(w) for w in words)

    grade = 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59
    return max(grade, 0.0)


@dataclass
class ScoredClause:
    order_idx: int
    buriedness: float
    impact_score: float
    # The components, kept rather than discarded. The UI can then explain *why*
    # a clause ranks where it does ("hard to read, and it points at 3 other
    # clauses"), and a wrong ranking can be debugged without re-running.
    position_signal: float
    reading_signal: float
    crossref_signal: float
    jargon_signal: float
    reading_grade: float


def _defined_terms(
    segments: list[Segment], analyses: dict[str, ClauseAnalysis]
) -> set[str]:
    """Collect the terms this policy defines in its own definition clauses.

    A clause that leans on defined terms cannot be understood where it sits -
    "Hospital", "Pre-existing Disease" and "Reasonable and Customary Charges"
    all mean something narrower than they appear. Detecting that requires
    knowing what THIS policy defines, so it is derived per document rather than
    from a fixed word list.
    """
    terms: set[str] = set()
    for seg in segments:
        analysis = analyses.get(str(seg.order_idx))
        if not analysis or analysis.clause_type != ClauseType.DEFINITION:
            continue
        # The defined term is the heading ("1.1 Hospital" -> "Hospital"), or
        # the words before "means" in the body.
        if seg.heading:
            head = seg.heading
            if seg.number:
                head = head[len(seg.number):]
            candidate = head.strip(" .-")
            if 2 < len(candidate) < 60:
                terms.add(candidate.lower())
        if m := re.search(r"^(.{2,60}?)\s+means\b", seg.text.strip(), re.I | re.M):
            terms.add(m.group(1).strip(" .-0123456789").lower())
    return {t for t in terms if t}


def score(
    segments: list[Segment], analyses: dict[str, ClauseAnalysis]
) -> dict[str, ScoredClause]:
    """Compute buriedness and impact for every analysed clause."""
    total = len(segments)
    terms = _defined_terms(segments, analyses)

    scored: dict[str, ScoredClause] = {}
    for rank, seg in enumerate(segments):
        key = str(seg.order_idx)
        analysis = analyses.get(key)
        if analysis is None:
            continue

        # 1. Position. Later is more buried - readers give up, and drafters put
        #    the unwelcome parts after the appealing ones.
        #
        #    Derived from the clause's rank in THIS list, not from
        #    `order_idx / len(segments)`. That earlier form silently assumed
        #    order_idx values always span 0..len-1; scoring any subset (as a
        #    unit test or a re-scoring pass does) produced ratios above 1 and
        #    impact scores in the hundreds. Rank is bounded by construction.
        position = rank / max(total - 1, 1)

        # 2. Reading difficulty. Grade 8 is ordinary prose, grade 20 is dense
        #    legal drafting; anything above 20 is not meaningfully worse to a
        #    reader who is already lost.
        grade = flesch_kincaid_grade(seg.text)
        reading = min(max((grade - 8.0) / 12.0, 0.0), 1.0)

        # 3. Cross-references. Four or more hops is treated as maximally
        #    obscure - past that point nobody is following the trail anyway.
        crossrefs = len(_RE_CROSSREF.findall(seg.text))
        crossref = min(crossrefs / 4.0, 1.0)

        # 4. Dependence on this policy's own defined terms.
        lowered = seg.text.lower()
        used = sum(1 for term in terms if term in lowered)
        jargon = min(used / 3.0, 1.0)

        buriedness = (
            W_POSITION * position
            + W_READING * reading
            + W_CROSSREF * crossref
            + W_JARGON * jargon
        )

        weight = TYPE_WEIGHT.get(analysis.clause_type, 0.5)
        blend = 0.5 * ((analysis.likelihood - 1) / 4) + 0.5 * ((analysis.severity - 1) / 4)
        impact = (
            100.0
            * weight
            * blend
            * (1 + BURIEDNESS_BOOST * buriedness)
            / (1 + BURIEDNESS_BOOST)
        )

        scored[key] = ScoredClause(
            order_idx=seg.order_idx,
            buriedness=round(buriedness, 4),
            impact_score=round(impact, 2),
            position_signal=round(position, 4),
            reading_signal=round(reading, 4),
            crossref_signal=round(crossref, 4),
            jargon_signal=round(jargon, 4),
            reading_grade=round(grade, 1),
        )
    return scored
