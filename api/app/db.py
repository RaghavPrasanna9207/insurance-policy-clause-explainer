"""Database engine and session management."""

from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    f"sqlite:///{settings.db_path}",
    # SQLite otherwise refuses use from any thread but the creating one.
    # FastAPI serves requests from a pool of threads, so this is required.
    connect_args={"check_same_thread": False},
)


class SchemaOutOfDate(RuntimeError):
    """The database on disk predates a change to the model definitions."""


def init_db() -> None:
    # No Alembic in v1: schema changes mean deleting data/app.db and
    # re-uploading. A migration tool would be ceremony for a single-user
    # local app with no data worth preserving yet.
    import app.models  # noqa: F401  - registers tables on SQLModel.metadata

    SQLModel.metadata.create_all(engine)
    check_schema()


def check_schema() -> None:
    """Fail loudly if the database predates a model change.

    `create_all` only creates tables that do not already exist - it never
    alters one. So adding a column to a model leaves an older database silently
    stale, and the mismatch only surfaces much later as
    `OperationalError: table clause has no column named number`, thrown from
    inside a background task where the message reaches nobody useful.

    This was found by a live-server smoke test, not by the suite, and could not
    have been found by the suite: the test fixtures call `drop_all` first, so
    tests always run against a freshly built schema and never against one that
    has aged. A whole class of bug is invisible to a test that rebuilds the
    world every time.

    Deliberately raises rather than dropping the tables itself. There is no
    migration path in v1, but silently discarding someone's uploaded policies
    is not this function's decision to make - so it reports exactly what is
    wrong and exactly how to fix it.
    """
    from sqlalchemy import inspect

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())

    problems: list[str] = []
    for name, table in SQLModel.metadata.tables.items():
        if name not in existing_tables:
            continue  # create_all just made it; it is current by definition
        on_disk = {col["name"] for col in inspector.get_columns(name)}
        missing = {c.name for c in table.columns} - on_disk
        if missing:
            problems.append(f"  table '{name}' is missing: {', '.join(sorted(missing))}")

    if problems:
        raise SchemaOutOfDate(
            "The database schema is out of date:\n"
            + "\n".join(problems)
            + f"\n\nThere are no migrations in v1. Delete {settings.db_path} and "
            "re-upload your documents.\nThe LLM cache is a separate file, so "
            "re-analysis will still be served from cache and take seconds."
        )


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    with Session(engine) as session:
        yield session
