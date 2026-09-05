from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from app.ml.model_manager import model_manager


class DetectionService:

    def predict(
        self,
        image: np.ndarray,
        model_type: str = "fls",
        confidence: float = 0.35,
    ) -> Dict[str, Any]:

        model = model_manager.load_model(model_type)

        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

        result = results[0]

        detections: List[Dict[str, Any]] = []

        if result.boxes is not None:
            for box in result.boxes:

                xyxy = box.xyxy[0].cpu().numpy().tolist()
                confidence_score = float(box.conf[0].cpu().item())
                class_id = int(box.cls[0].cpu().item())

                class_name = model.names[class_id]

                x1, y1, x2, y2 = xyxy

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence_score,
                    "bbox": {
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                    },
                })

        height, width = image.shape[:2]

        return {
            "model": model_type,
            "image": {
                "width": width,
                "height": height,
            },
            "detection_count": len(detections),
            "detections": detections,
        }


detection_service = DetectionService()