from fastapi import APIRouter

from app.api.v1 import chat, ingest, layers, markets, reports, signals, training

router = APIRouter()
router.include_router(signals.router)
router.include_router(layers.router)
router.include_router(markets.router)
router.include_router(reports.router)
router.include_router(training.router)
router.include_router(chat.router)
router.include_router(ingest.router)
