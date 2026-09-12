import uuid
from sqlalchemy import Column, String, Text, DateTime, Enum, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.database import Base

class ReviewDecision(enum.Enum):
    CONFIRM = "CONFIRM"
    REJECT = "REJECT"
    RECLASSIFY = "RECLASSIFY"

class TargetReview(Base):
    __tablename__ = "target_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    target_id = Column(UUID(as_uuid=True), ForeignKey("targets.id"))
    reviewer_id = Column(UUID(as_uuid=True), ForeignKey("users.id"))
    decision = Column(Enum(ReviewDecision), nullable=False)
    final_class = Column(String, nullable=True)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    target = relationship("Target", back_populates="target_reviews")
    reviewer = relationship("User", back_populates="target_reviews")
