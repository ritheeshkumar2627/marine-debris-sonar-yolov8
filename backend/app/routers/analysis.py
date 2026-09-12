from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import Any, Dict

from app.services.ai_service import ai_service

router = APIRouter(tags=["AI Analysis"])

@router.post("/surveys/{id}/analyze")
async def trigger_analysis(id: str):
    return {"message": f"Trigger analysis for survey {id} placeholder"}

@router.get("/analysis/{id}")
async def get_analysis_run(id: str):
    return {"message": f"Get analysis run {id} placeholder"}

@router.get("/analysis/{id}/status")
async def get_analysis_status(id: str):
    return {"message": f"Get analysis status {id} placeholder"}

@router.post("/analysis/test-image", response_model=Dict[str, Any])
async def test_image_analysis(
    file: UploadFile = File(...),
    model_type: str = "fls",
    confidence: float = 0.35
):
    """
    Test endpoint to upload an image and get detection results directly 
    from the connected ML service without saving to the DB.
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    image_bytes = await file.read()
    
    try:
        results = await ai_service.analyze_image(
            image_bytes=image_bytes,
            model_type=model_type,
            confidence=confidence
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
