import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.config import settings

logger = logging.getLogger(__name__)


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
    yield
    # ── shutdown ──────────────────────────────────────────────────────────
    # nothing to clean up currently


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


@app.get("/health", tags=["meta"])
async def health():
    from app.inference.pipeline import is_model_loaded
    return {
        "status": "ok",
        "model_loaded": is_model_loaded(),
    }
