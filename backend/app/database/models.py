from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from backend.app.database.session import Base

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    text = Column(Text, nullable=False)
    raw_eml = Column(Text, nullable=True)
    prediction_label = Column(String(50), nullable=False)
    risk_score = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False)
    processing_time = Column(Float, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships with cascade delete
    rules_triggered = relationship("RuleTrigger", back_populates="analysis", cascade="all, delete-orphan")
    url_findings = relationship("URLFinding", back_populates="analysis", cascade="all, delete-orphan")
    llm_report = relationship("LLMReport", back_populates="analysis", uselist=False, cascade="all, delete-orphan")

class RuleTrigger(Base):
    __tablename__ = "rules_triggered"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    rule_name = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    confidence = Column(Float, nullable=False)
    reason = Column(Text, nullable=False)

    analysis = relationship("Analysis", back_populates="rules_triggered")

class URLFinding(Base):
    __tablename__ = "url_findings"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    flags = Column(Text, nullable=True)  # Comma separated tags/JSON
    entropy = Column(Float, nullable=False)
    is_suspicious = Column(Boolean, default=False)

    analysis = relationship("Analysis", back_populates="url_findings")

class LLMReport(Base):
    __tablename__ = "llm_reports"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id", ondelete="CASCADE"), unique=True, nullable=False)
    threat_type = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    summary = Column(Text, nullable=False)
    indicators = Column(Text, nullable=True)  # Comma separated list of indicators
    recommendations = Column(Text, nullable=False)
    executive_summary = Column(Text, nullable=False)

    analysis = relationship("Analysis", back_populates="llm_report")
