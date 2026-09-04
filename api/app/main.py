"""FastAPI application entry point."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db import init_db
from app.llm import cache, client

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs once on startup, and again (after the yield) on shutdown.

    Warming the model here pays the ~27s cold load before any user is waiting,
    rather than during their first upload where it looks like a hang.
    """
    init_db()
    log.info("database ready at %s", settings.db_path)
    await client.warm()
    yield


app = FastAPI(
    title="Insurance Policy Clause Explainer",
    description="Finds the clauses in your health policy that could get a claim denied.",
    version="0.1.0",
    lifespan=lifespan,
)

# The Vite dev server runs on a different port, so the browser treats API calls
# as cross-origin and blocks them without this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    """Reports whether the pieces this app depends on are actually up.

    Deliberately does more than return {"ok": true}: the most common failure
    here is Ollama running but the expected model not pulled, which this
    surfaces explicitly instead of failing later inside the pipeline.
    """
    llm = await client.health()
    return {"status": "ok" if llm["model_available"] else "degraded",
            "llm": llm,
            "cache": cache.stats()}
