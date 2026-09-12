import httpx
from typing import Dict, Any

from app.config import settings

class AiService:
    def __init__(self):
        self.ml_service_url = settings.ML_SERVICE_URL

    async def analyze_image(self, image_bytes: bytes, model_type: str = "fls", confidence: float = 0.35) -> Dict[str, Any]:
        url = f"{self.ml_service_url}/api/v1/detection/image"
        
        files = {
            'file': ('image.png', image_bytes, 'image/png')
        }
        data = {
            'model': model_type,
            'confidence': str(confidence)
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(url, files=files, data=data, timeout=30.0)
            
            if response.status_code != 200:
                raise Exception(f"ML Service error: {response.status_code} {response.text}")
                
            return response.json()

ai_service = AiService()
