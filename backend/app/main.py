from fastapi import FastAPI
from app.config import settings

from app.routers import auth, surveys, sonar, analysis, detections, targets, reviews

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Backend API for Marine Debris Sonar Analysis Platform"
)

app.include_router(auth.router)
app.include_router(surveys.router)
app.include_router(sonar.router)
app.include_router(analysis.router)
app.include_router(detections.router)
app.include_router(targets.router)
app.include_router(reviews.router)

@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME}"}
