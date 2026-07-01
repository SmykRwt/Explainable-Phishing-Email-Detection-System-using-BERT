import time
import logging
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from typing import List, Optional

from backend.app.database.session import get_db
from backend.app.database.models import Analysis, RuleTrigger, URLFinding, LLMReport
from backend.app.schemas.analysis import AnalysisRequest, AnalysisResponse
from backend.app.core.dependencies import get_predictor, get_llm_analyst
from backend.app.services.email_parser import EmailParser
from backend.app.services.url_analyzer import URLAnalyzer
from backend.app.services.header_analyzer import HeaderAnalyzer
from backend.app.services.threat_scorer import ThreatScorer
from backend.app.services.ocr_service import OCRService
from backend.app.rules.engine import RuleEngine

logger = logging.getLogger("phishing_platform")
router = APIRouter()

async def run_pipeline_and_save(
    text_content: str,
    raw_eml: Optional[str],
    is_eml: bool,
    sender: str,
    reply_to: str,
    subject: str,
    urls: List[str],
    attachments: List[dict],
    headers: dict,
    db: AsyncSession,
    predictor,
    llm_analyst
) -> Analysis:
    """Helper that runs threat scoring, OCR or email text analysis, queries the LLM and writes to DB."""
    t0 = time.time()

    # 1. Parse/Analyze URLs
    url_findings_list = []
    for url in urls[:5]:  # Analyze up to 5 URLs to keep it fast
        analysis_res = URLAnalyzer.analyze_url(url)
        url_findings_list.append(analysis_res)

    # 2. Analyze Headers
    header_findings = HeaderAnalyzer.analyze_headers(headers, sender, reply_to)

    # 3. Compile Parsed Data for Rules Engine
    email_data = {
        "body": text_content,
        "subject": subject,
        "urls": urls,
        "attachments": attachments,
        "sender": sender,
        "reply_to": reply_to
    }
    triggered_rules = RuleEngine.evaluate_rules(email_data)

    # 4. Predict probabilities using ML models
    # Combine subject and body for BERT classification context
    bert_context = f"Subject: {subject}\n\n{text_content}" if subject else text_content
    probs = predictor.predict_bert(bert_context)
    spam_prob = float(probs[1]) * 100.0  # Percentage

    # 5. Composite Risk Score
    scoring_result = ThreatScorer.compute_risk_score(
        bert_spam_prob=spam_prob,
        rules=triggered_rules,
        url_findings=url_findings_list,
        header_findings=header_findings
    )
    
    risk_score = scoring_result["risk_score"]
    verdict = scoring_result["verdict"]

    # 6. GenAI Explanation Report
    llm_report_dict = await llm_analyst.analyze_threat(
        verdict=verdict,
        risk_score=risk_score,
        rules=[r.model_dump() for r in triggered_rules],
        urls=url_findings_list,
        headers=header_findings,
        email_text=text_content
    )

    processing_time = time.time() - t0

    # 7. Write findings to PostgreSQL/SQLite
    db_analysis = Analysis(
        text=text_content,
        raw_eml=raw_eml,
        prediction_label=verdict,
        risk_score=risk_score,
        model_version="DistilBERT-v1 + Classical Ensemble",
        processing_time=processing_time
    )
    db.add(db_analysis)
    await db.flush()  # Populates db_analysis.id

    # Add rules
    for r in triggered_rules:
        db_rule = RuleTrigger(
            analysis_id=db_analysis.id,
            rule_name=r.rule_name,
            severity=r.severity,
            confidence=r.confidence,
            reason=r.reason
        )
        db.add(db_rule)

    # Add URLs
    for u in url_findings_list:
        db_url = URLFinding(
            analysis_id=db_analysis.id,
            url=u["url"],
            flags=",".join(u["flags"]) if u["flags"] else "",
            entropy=u["entropy"],
            is_suspicious=u["is_suspicious"]
        )
        db.add(db_url)

    # Add LLM report
    db_llm = LLMReport(
        analysis_id=db_analysis.id,
        threat_type=llm_report_dict["threat_type"],
        severity=llm_report_dict["severity"],
        summary=llm_report_dict["summary"],
        indicators=",".join(llm_report_dict["indicators"]) if isinstance(llm_report_dict["indicators"], list) else str(llm_report_dict["indicators"]),
        recommendations=llm_report_dict["recommendations"],
        executive_summary=llm_report_dict["executive_summary"]
    )
    db.add(db_llm)

    await db.commit()
    
    # Reload with relationships
    stmt = (
        select(Analysis)
        .options(
            selectinload(Analysis.rules_triggered),
            selectinload(Analysis.url_findings),
            selectinload(Analysis.llm_report)
        )
        .where(Analysis.id == db_analysis.id)
    )
    res = await db.execute(stmt)
    return res.scalar_one()


@router.post("/email", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_email_json(
    request: AnalysisRequest,
    db: AsyncSession = Depends(get_db),
    predictor = Depends(get_predictor),
    llm_analyst = Depends(get_llm_analyst)
):
    """Analyzes raw email/text submitted in a JSON payload."""
    parsed = EmailParser.parse_raw_text(request.text)
    
    # Run pipeline
    return await run_pipeline_and_save(
        text_content=parsed["body"],
        raw_eml=request.raw_eml,
        is_eml=parsed["is_eml"],
        sender=parsed["sender"],
        reply_to=parsed["reply_to"],
        subject=parsed["subject"],
        urls=parsed["urls"],
        attachments=parsed["attachments"],
        headers=parsed["headers"],
        db=db,
        predictor=predictor,
        llm_analyst=llm_analyst
    )


@router.post("/email/upload", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_email_upload(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    predictor = Depends(get_predictor),
    llm_analyst = Depends(get_llm_analyst)
):
    """Analyzes an uploaded .eml file."""
    content_bytes = await file.read()
    content_str = content_bytes.decode(errors="ignore")
    
    parsed = EmailParser.parse_eml(content_str)
    
    return await run_pipeline_and_save(
        text_content=parsed["body"],
        raw_eml=content_str,
        is_eml=True,
        sender=parsed["sender"],
        reply_to=parsed["reply_to"],
        subject=parsed["subject"],
        urls=parsed["urls"],
        attachments=parsed["attachments"],
        headers=parsed["headers"],
        db=db,
        predictor=predictor,
        llm_analyst=llm_analyst
    )


@router.post("/screenshot", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def analyze_screenshot(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    predictor = Depends(get_predictor),
    llm_analyst = Depends(get_llm_analyst)
):
    """Extracts text from screenshot upload using Tesseract OCR and performs phishing analysis."""
    image_bytes = await file.read()
    extracted_text = OCRService.extract_text_from_bytes(image_bytes)
    
    if not extracted_text or extracted_text.startswith("[OCR"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to perform OCR extraction: {extracted_text}"
        )
        
    parsed = EmailParser.parse_raw_text(extracted_text)
    
    return await run_pipeline_and_save(
        text_content=parsed["body"],
        raw_eml=f"[Extracted via OCR]\n{extracted_text}",
        is_eml=False,
        sender=parsed["sender"],
        reply_to=parsed["reply_to"],
        subject=parsed["subject"],
        urls=parsed["urls"],
        attachments=[],
        headers=parsed.get("headers", {}),
        db=db,
        predictor=predictor,
        llm_analyst=llm_analyst
    )


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(
    analysis_id: int,
    db: AsyncSession = Depends(get_db)
):
    """Retrieves detailed results for a specific analysis record."""
    stmt = (
        select(Analysis)
        .options(
            selectinload(Analysis.rules_triggered),
            selectinload(Analysis.url_findings),
            selectinload(Analysis.llm_report)
        )
        .where(Analysis.id == analysis_id)
    )
    res = await db.execute(stmt)
    analysis = res.scalar_one_or_none()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID {analysis_id} not found."
        )
    return analysis
