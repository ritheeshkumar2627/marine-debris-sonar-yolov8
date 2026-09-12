import uuid
from sqlalchemy import Column, String, Text, DateTime, Enum, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.database import Base

class ModelType(enum.Enum):
    DETECTOR = "DETECTOR"
    ANOMALY = "ANOMALY"
    SEGMENTATION = "SEGMENTATION"
    TRACKER = "TRACKER"

class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String, nullable=False)
    version = Column(String, nullable=False)
    model_type = Column(Enum(ModelType), nullable=False)
    weights_path = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis_runs = relationship("AnalysisRun", back_populates="model_version")
