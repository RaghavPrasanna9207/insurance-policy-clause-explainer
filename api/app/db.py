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


def init_db() -> None:
    # No Alembic in v1: schema changes mean deleting data/app.db and
    # re-uploading. A migration tool would be ceremony for a single-user
    # local app with no data worth preserving yet.
    import app.models  # noqa: F401  - registers tables on SQLModel.metadata

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    """FastAPI dependency: one session per request, always closed."""
    with Session(engine) as session:
        yield session
