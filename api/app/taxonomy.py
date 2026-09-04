"""The shared vocabulary of the system.

These enums are the contract between three things that must never drift apart:
the database columns, the JSON schemas sent to the model, and the frontend
filters. Defining them once and deriving the schema `enum` lists from them (via
`values()`) means a new clause type cannot be added to one layer and forgotten
in another.
"""

from enum import StrEnum


class ClauseType(StrEnum):
    """What role a clause plays in deciding a claim.

    Ordered roughly by how directly each one can cost a policyholder money.
    """

    # Says what the policy DOES pay for.
    COVERAGE = "coverage"
    # Says what it will NEVER pay for. The single biggest source of denials.
    EXCLUSION = "exclusion"
    # An obligation on the policyholder; failing it can void an otherwise valid
    # claim (e.g. "notify us within 24 hours of hospitalisation").
    CONDITION = "condition"
    # Caps a payout below the sum insured (e.g. "room rent limited to 1% per
    # day"). Rarely denies a claim outright, but quietly shrinks it.
    SUB_LIMIT = "sub_limit"
    # Coverage that only begins after some time has elapsed (e.g. pre-existing
    # disease covered only after 36 months).
    WAITING_PERIOD = "waiting_period"
    # Defines a term used elsewhere. Low direct impact, high indirect impact:
    # a narrow definition of "hospital" can silently gut a coverage clause.
    DEFINITION = "definition"
    # Administrative machinery: how to file, renew, cancel, port.
    PROCEDURAL = "procedural"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


class Verdict(StrEnum):
    """The scenario simulator's answer."""

    COVERED = "covered"
    NOT_COVERED = "not_covered"
    # Covered only if some condition holds that the scenario didn't specify.
    CONDITIONAL = "conditional"
    # The document genuinely does not address this. A first-class answer, not a
    # failure: in this domain, guessing is worse than admitting ignorance.
    INSUFFICIENT_INFORMATION = "insufficient_information"

    @classmethod
    def values(cls) -> list[str]:
        return [m.value for m in cls]


class DocStatus(StrEnum):
    """Where a document is in the pipeline, for the polling UI."""

    PENDING = "pending"
    INGESTING = "ingesting"
    SEGMENTING = "segmenting"
    ANALYZING = "analyzing"
    SCORING = "scoring"
    READY = "ready"
    FAILED = "failed"
