from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.report import IntuOneReport
from app.schemas.report import IntuOneReportCreate, IntuOneReportRead

router = APIRouter(prefix="/reports", tags=["reports"])


@router.post("/", response_model=IntuOneReportRead, status_code=201)
async def create_report(payload: IntuOneReportCreate, db: AsyncSession = Depends(get_db)):
    report = IntuOneReport(**payload.model_dump())
    db.add(report)
    await db.commit()
    await db.refresh(report)
    return report


@router.get("/", response_model=list[IntuOneReportRead])
async def list_reports(
    topic: str | None = None,
    time_window: str | None = None,
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    q = select(IntuOneReport)
    if topic:
        q = q.where(IntuOneReport.topic == topic)
    if time_window:
        q = q.where(IntuOneReport.time_window == time_window)
    q = q.order_by(IntuOneReport.generated_at.desc()).limit(limit).offset(offset)
    result = await db.execute(q)
    return result.scalars().all()


@router.get("/latest", response_model=IntuOneReportRead)
async def get_latest_report(topic: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(IntuOneReport)
        .where(IntuOneReport.topic == topic)
        .order_by(IntuOneReport.generated_at.desc())
        .limit(1)
    )
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="No report found for this topic")
    return report


@router.get("/{report_id}", response_model=IntuOneReportRead)
async def get_report(report_id: UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(IntuOneReport).where(IntuOneReport.id == report_id))
    report = result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
