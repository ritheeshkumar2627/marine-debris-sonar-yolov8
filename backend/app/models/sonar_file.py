import uuid
from sqlalchemy import Column, String, Text, BigInteger, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship

from app.database import Base

class SonarFile(Base):
    __tablename__ = "sonar_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    survey_id = Column(UUID(as_uuid=True), ForeignKey("surveys.id"))
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    storage_path = Column(Text, nullable=False)
    file_size = Column(BigInteger, nullable=True)
    metadata_json = Column("metadata", JSONB, nullable=True)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    survey = relationship("Survey", back_populates="sonar_files")
    frames = relationship("Frame", back_populates="sonar_file")
