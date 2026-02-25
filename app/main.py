import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import router as v1_router
from app.config import settings

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ───────────────────────────────────────────────────────────
    if settings.LOAD_MODEL_ON_STARTUP:
        from app.inference.pipeline import load_model
        logger.info("Loading IntuOne fine-tuned model …")
        load_model(
            adapter_path=settings.ADAPTER_PATH,
            base_model=settings.BASE_MODEL_NAME,
        )
        logger.info("Model ready.")

    # Start background data ingestion loop
    from app.services.ingestion import ingestion_loop
    _ingest_task = asyncio.create_task(ingestion_loop())
    logger.info("Ingestion background loop started.")

    yield

    # ── shutdown ──────────────────────────────────────────────────────────
    _ingest_task.cancel()
    try:
        await _ingest_task
    except asyncio.CancelledError:
        pass


app = FastAPI(
    title="ParallelLines — IntuOne Perception Engine",
    description=(
        "Multi-layer internet signal analysis. "
        "Six layers (market, social, news, sentiment, geopolitical, synthesis) "
        "feed into IntuOne to produce real-time intelligence briefings."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(v1_router, prefix=settings.API_V1_PREFIX)

# Serve the frontend (chat + dashboard)
if _STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


@app.get("/health", tags=["meta"])
async def health():
    from app.inference.pipeline import is_model_loaded
    from app.services.ingestion import get_stats
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
        "ingestion": get_stats(),
    }
