from fastapi import APIRouter
from typing import Optional

router = APIRouter(tags=["Targets"])

@router.get("/surveys/{id}/targets")
async def get_survey_targets(id: str):
    return {"message": f"Get targets for survey {id} placeholder"}

@router.get("/surveys/{id}/targets/geojson")
async def get_survey_targets_geojson(id: str):
    return {"type": "FeatureCollection", "features": [], "message": f"GeoJSON placeholder for survey {id}"}

@router.get("/targets")
async def get_targets(priority: Optional[str] = None):
    return {"message": f"List targets, priority={priority} placeholder"}

@router.get("/targets/{id}")
async def get_target(id: str):
    return {"message": f"Get target {id} placeholder"}
