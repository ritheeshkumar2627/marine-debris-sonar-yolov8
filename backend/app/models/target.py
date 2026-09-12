import uuid
from sqlalchemy import Column, String, Float, Double, Integer, DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry
import enum

from app.database import Base

class Priority(enum.Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class TargetStatus(enum.Enum):
    PENDING = "PENDING"
    VERIFIED = "VERIFIED"
    REJECTED = "REJECTED"

class Target(Base):
    __tablename__ = "targets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"))
    target_type = Column(String, nullable=True)
    confidence_score = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=True)
    shadow_score = Column(Float, nullable=True)
    priority = Column(Enum(Priority), default=Priority.LOW)
    risk_score = Column(Float, nullable=True)
    latitude = Column(Double, nullable=True)
    longitude = Column(Double, nullable=True)
    
    # PostGIS Point for the consolidated target location
    geometry = Column(Geometry(geometry_type='POINT', srid=4326), nullable=True)
    
    estimated_size_m = Column(Float, nullable=True)
    detection_count = Column(Integer, default=1)
    status = Column(Enum(TargetStatus), default=TargetStatus.PENDING)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    survey = relationship("Survey", back_populates="targets")
    detections = relationship("Detection", back_populates="target")
    target_reviews = relationship("TargetReview", back_populates="target")
