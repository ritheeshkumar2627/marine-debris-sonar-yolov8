# Marine Debris Sonar ML Service

FastAPI service for marine debris detection using YOLOv8.

## Currently Supported

- FLS (Forward-Looking Sonar)

SSS is not implemented yet.

## Start Service

From the `ml-service` directory:

```powershell
python -m uvicorn app.main:app --reload --port 8000