from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.detection import router as detection_router


app = FastAPI(
    title="Marine Debris Sonar ML Service",
    description="AI-powered marine debris detection using sonar imagery.",
    version="1.0.0",
)


app.include_router(health_router)
app.include_router(detection_router)