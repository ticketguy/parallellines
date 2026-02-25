from fastapi import APIRouter

from app.api.v1 import layers, markets, reports, signals

router = APIRouter()
router.include_router(signals.router)
router.include_router(layers.router)
router.include_router(markets.router)
router.include_router(reports.router)
