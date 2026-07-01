from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime

class RuleResult(BaseModel):
    rule_name: str
    severity: str  # Low, Medium, High, Critical
    confidence: float
    reason: str

    class Config:
        from_attributes = True

class URLFindingBase(BaseModel):
    url: str
    flags: List[str]
    entropy: float
    is_suspicious: bool

    class Config:
        from_attributes = True

    @field_validator("flags", mode="before")
    @classmethod
    def parse_flags(cls, v):
        if isinstance(v, str):
            if not v:
                return []
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

class LLMReportBase(BaseModel):
    threat_type: str
    severity: str
    summary: str
    indicators: List[str]
    recommendations: str
    executive_summary: str

    class Config:
        from_attributes = True

    @field_validator("indicators", mode="before")
    @classmethod
    def parse_indicators(cls, v):
        if isinstance(v, str):
            if not v:
                return []
            return [x.strip() for x in v.split(",") if x.strip()]
        return v

class AnalysisRequest(BaseModel):
    text: str
    raw_eml: Optional[str] = None

class AnalysisResponse(BaseModel):
    id: int
    text: str
    raw_eml: Optional[str]
    prediction_label: str
    risk_score: float
    model_version: str
    processing_time: float
    created_at: datetime
    rules_triggered: List[RuleResult] = []
    url_findings: List[URLFindingBase] = []
    llm_report: Optional[LLMReportBase] = None

    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_analyses: int
    avg_risk_score: float
    risk_distribution: Dict[str, int]  # e.g., {"Low": 10, "Medium": 5, "High": 8, "Critical": 2}
    threat_categories: Dict[str, int]  # e.g., {"Credential Harvesting": 5, "Urgency": 12, ...}
    processing_latency: float
    top_suspicious_domains: List[Dict[str, int]]  # e.g., [{"domain.com": 5}, ...]
    recent_analyses: List[AnalysisResponse]
