import httpx
from typing import Dict, Any

from app.config import settings

CLASS_MAPPING: Dict[Any, str] = {
    1: "fishnet",
    8: "pipe",
    9: "cylinder",
    "bottle": "fishnet",
    "shampoo-bottle": "pipe",
    "standing-bottle": "cylinder",
}

class AiService:
    def __init__(self):
        self.ml_service_url = settings.ML_SERVICE_URL

    async def analyze_image(self, image_bytes: bytes, model_type: str = "sss", confidence: float = 0.35) -> Dict[str, Any]:
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
                
            res = response.json()

            if "detections" in res and isinstance(res["detections"], list):
                for det in res["detections"]:
                    c_id = det.get("class_id")
                    c_name = det.get("class_name")
                    if c_id in CLASS_MAPPING:
                        det["class_name"] = CLASS_MAPPING[c_id]
                    elif c_name in CLASS_MAPPING:
                        det["class_name"] = CLASS_MAPPING[c_name]

            return res

ai_service = AiService()
