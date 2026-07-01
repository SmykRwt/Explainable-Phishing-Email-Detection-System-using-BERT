from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func, desc
from sqlalchemy.orm import selectinload
from typing import List, Dict, Any
from urllib.parse import urlparse

from backend.app.database.session import get_db
from backend.app.database.models import Analysis, RuleTrigger, URLFinding
from backend.app.schemas.analysis import AnalysisResponse, DashboardStats

router = APIRouter()

@router.get("/history", response_model=List[AnalysisResponse])
async def get_history(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves list of past analyses ordered by date (newest first) with offset paging."""
    stmt = (
        select(Analysis)
        .options(
            selectinload(Analysis.rules_triggered),
            selectinload(Analysis.url_findings),
            selectinload(Analysis.llm_report)
        )
        .order_by(desc(Analysis.created_at))
        .limit(limit)
        .offset(offset)
    )
    res = await db.execute(stmt)
    return res.scalars().all()


@router.get("/dashboard", response_model=DashboardStats)
async def get_dashboard_stats(
    db: AsyncSession = Depends(get_db)
):
    """Aggregates metrics and statistics for the main dashboard views."""
    # 1. Total count
    total_stmt = select(func.count(Analysis.id))
    total_res = await db.execute(total_stmt)
    total_count = total_res.scalar() or 0

    if total_count == 0:
        # Return empty dashboard structure
        return DashboardStats(
            total_analyses=0,
            avg_risk_score=0.0,
            risk_distribution={},
            threat_categories={},
            processing_latency=0.0,
            top_suspicious_domains=[],
            recent_analyses=[]
        )

    # 2. Avg risk score
    avg_score_stmt = select(func.avg(Analysis.risk_score))
    avg_score_res = await db.execute(avg_score_stmt)
    avg_score = float(avg_score_res.scalar() or 0.0)

    # 3. Avg processing latency
    avg_lat_stmt = select(func.avg(Analysis.processing_time))
    avg_lat_res = await db.execute(avg_lat_stmt)
    avg_latency = float(avg_lat_res.scalar() or 0.0)

    # 4. Risk distribution (Safe vs Suspicious vs Phishing)
    dist_stmt = select(Analysis.prediction_label, func.count(Analysis.id)).group_by(Analysis.prediction_label)
    dist_res = await db.execute(dist_stmt)
    risk_distribution = {row[0]: row[1] for row in dist_res.all()}

    # 5. Threat categories (rules)
    rules_stmt = select(RuleTrigger.rule_name, func.count(RuleTrigger.id)).group_by(RuleTrigger.rule_name)
    rules_res = await db.execute(rules_stmt)
    threat_categories = {row[0]: row[1] for row in rules_res.all()}

    # 6. Top suspicious domains
    url_stmt = (
        select(URLFinding.url)
        .where(URLFinding.is_suspicious == True)
        .limit(100) # pull recent 100 suspicious urls to parse domains
    )
    url_res = await db.execute(url_stmt)
    domains_counter = {}
    for row in url_res.scalars().all():
        try:
            parsed = urlparse(row)
            domain = parsed.netloc or row
            if ":" in domain:
                domain = domain.split(":")[0]
            domains_counter[domain] = domains_counter.get(domain, 0) + 1
        except Exception:
            pass
            
    # Sort and format top domains
    sorted_domains = sorted(domains_counter.items(), key=lambda x: -x[1])[:5]
    top_domains = [{item[0]: item[1]} for item in sorted_domains]

    # 7. Recent 5 analyses
    recent_stmt = (
        select(Analysis)
        .options(
            selectinload(Analysis.rules_triggered),
            selectinload(Analysis.url_findings),
            selectinload(Analysis.llm_report)
        )
        .order_by(desc(Analysis.created_at))
        .limit(5)
    )
    recent_res = await db.execute(recent_stmt)
    recent_analyses = recent_res.scalars().all()

    return DashboardStats(
        total_analyses=total_count,
        avg_risk_score=round(avg_score, 2),
        risk_distribution=risk_distribution,
        threat_categories=threat_categories,
        processing_latency=round(avg_latency, 3),
        top_suspicious_domains=top_domains,
        recent_analyses=recent_analyses
    )


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """Exposes backend performance indicators."""
    total_res = await db.execute(select(func.count(Analysis.id)))
    total = total_res.scalar() or 0
    return {
        "status": "online",
        "total_records_processed": total,
        "api_version": "v1.0"
    }


@router.get("/health")
def health_check():
    """Confirms application router health."""
    return {"status": "healthy"}
