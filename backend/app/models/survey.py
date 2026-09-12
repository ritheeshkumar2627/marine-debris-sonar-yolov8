import uuid
from sqlalchemy import Column, String, Text, DateTime, Enum, Float, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.database import Base

class PlatformType(enum.Enum):
    VESSEL = "VESSEL"
    AUV = "AUV"
    USV = "USV"
    OTHER = "OTHER"

class SurveyStatus(enum.Enum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class Survey(Base):
    __tablename__ = "surveys"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    location_name = Column(String, nullable=True)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    platform_type = Column(Enum(PlatformType), nullable=True)
    sonar_frequency = Column(Float, nullable=True)
    status = Column(Enum(SurveyStatus), default=SurveyStatus.UPLOADED, nullable=False)
    
    created_by = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    creator = relationship("User", back_populates="surveys")
    sonar_files = relationship("SonarFile", back_populates="survey")
    analysis_runs = relationship("AnalysisRun", back_populates="survey")
    targets = relationship("Target", back_populates="survey")
