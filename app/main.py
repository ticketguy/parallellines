from fastapi import FastAPI

from app.api.v1.router import router as v1_router
from app.config import settings

app = FastAPI(
    title="ParallelLines — IntuOne Perception Engine",
    description=(
        "Multi-layer internet signal analysis. "
        "Six layers (market, social, news, sentiment, geopolitical, synthesis) "
        "feed into the IntuOne orchestrator to produce directional intelligence reports."
    ),
    version="0.1.0",
)

app.include_router(v1_router, prefix=settings.API_V1_PREFIX)


@app.get("/health", tags=["meta"])
async def health():
    return {"status": "ok"}
