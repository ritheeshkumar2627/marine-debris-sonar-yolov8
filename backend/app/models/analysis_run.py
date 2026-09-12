import uuid
from sqlalchemy import Column, Integer, BigInteger, Text, DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.database import Base

class RunStatus(enum.Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"))
    model_version_id = Column(UUID(as_uuid=True), ForeignKey("model_versions.id"))
    status = Column(Enum(RunStatus), default=RunStatus.QUEUED, nullable=False)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    frames_processed = Column(Integer, default=0)
    total_detections = Column(Integer, default=0)
    processing_time_ms = Column(BigInteger, nullable=True)
    error_message = Column(Text, nullable=True)

    survey = relationship("Survey", back_populates="analysis_runs")
    model_version = relationship("ModelVersion", back_populates="analysis_runs")
    detections = relationship("Detection", back_populates="analysis_run")
