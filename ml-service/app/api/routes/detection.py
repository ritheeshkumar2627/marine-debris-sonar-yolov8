import cv2
import numpy as np

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas.detection import DetectionResponse
from app.services.detection_service import detection_service


router = APIRouter(
    prefix="/api/v1/detection",
    tags=["Detection"],
)


@router.post(
    "/image",
    response_model=DetectionResponse,
)
async def detect_image(
    file: UploadFile = File(...),
    model: str = Form("fls"),
    confidence: float = Form(0.35),
):

    if model not in {"fls", "sss"}:
        raise HTTPException(
            status_code=400,
            detail="model must be either 'fls' or 'sss'",
        )

    if not 0.0 <= confidence <= 1.0:
        raise HTTPException(
            status_code=400,
            detail="confidence must be between 0 and 1",
        )

    contents = await file.read()

    image_array = np.frombuffer(contents, dtype=np.uint8)

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    try:
        result = detection_service.predict(
            image=image,
            model_type=model,
            confidence=confidence,
        )

        return result

    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        )