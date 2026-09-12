import uuid
from sqlalchemy import Column, String, Float, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base

class Detection(Base):
    __tablename__ = "detections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    analysis_run_id = Column(UUID(as_uuid=True), ForeignKey("analysis_runs.id"))
    frame_id = Column(UUID(as_uuid=True), ForeignKey("frames.id"))
    class_name = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    anomaly_score = Column(Float, nullable=True)
    shadow_score = Column(Float, nullable=True)
    bbox_x = Column(Float, nullable=True)
    bbox_y = Column(Float, nullable=True)
    bbox_width = Column(Float, nullable=True)
    bbox_height = Column(Float, nullable=True)
    segmentation_path = Column(Text, nullable=True)
    estimated_size_m = Column(Float, nullable=True)
    
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    analysis_run = relationship("AnalysisRun", back_populates="detections")
    frame = relationship("Frame", back_populates="detections")
    target = relationship("Target", back_populates="detections")
