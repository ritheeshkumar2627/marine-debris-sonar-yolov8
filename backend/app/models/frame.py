import uuid
from sqlalchemy import Column, Integer, Text, Double, Float, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from geoalchemy2 import Geometry

from app.database import Base

class Frame(Base):
    __tablename__ = "frames"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    sonar_file_id = Column(UUID(as_uuid=True), ForeignKey("sonar_files.id"))
    frame_number = Column(Integer, nullable=False)
    image_path = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=True)
    latitude = Column(Double, nullable=True)
    longitude = Column(Double, nullable=True)
    heading = Column(Float, nullable=True)
    altitude = Column(Float, nullable=True)
    range_m = Column(Float, nullable=True)
    
    # 4326 is WGS 84 (standard GPS coordinates)
    geometry = Column(Geometry(geometry_type='POINT', srid=4326), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sonar_file = relationship("SonarFile", back_populates="frames")
    detections = relationship("Detection", back_populates="frame")
