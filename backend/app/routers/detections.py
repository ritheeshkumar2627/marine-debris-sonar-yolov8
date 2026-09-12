from fastapi import APIRouter

router = APIRouter(tags=["Detections"])

@router.get("/surveys/{id}/detections")
async def get_survey_detections(id: str):
    return {"message": f"Get detections for survey {id} placeholder"}

@router.get("/detections/{id}")
async def get_detection(id: str):
    return {"message": f"Get detection {id} placeholder"}
