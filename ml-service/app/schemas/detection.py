from typing import List

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class ImageInfo(BaseModel):
    width: int
    height: int


class DetectionResponse(BaseModel):
    model: str
    image: ImageInfo
    detection_count: int
    detections: List[Detection]