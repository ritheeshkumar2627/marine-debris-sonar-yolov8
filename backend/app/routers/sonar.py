from fastapi import APIRouter

router = APIRouter(tags=["Sonar"])

@router.post("/surveys/{id}/upload")
async def upload_sonar_file(id: str):
    return {"message": f"Upload sonar file for survey {id} placeholder"}

@router.get("/surveys/{id}/files")
async def get_sonar_files(id: str):
    return {"message": f"List sonar files for survey {id} placeholder"}

@router.get("/files/{id}/frames")
async def get_frames(id: str):
    return {"message": f"List frames for file {id} placeholder"}
