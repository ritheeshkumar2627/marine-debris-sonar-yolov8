from pathlib import Path
from typing import Dict

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = BASE_DIR / "models"


MODEL_PATHS: Dict[str, Path] = {
    "fls": MODELS_DIR / "fls" / "best.pt",
    "sss": MODELS_DIR / "sss" / "best.pt",
}


class ModelManager:
    def __init__(self):
        self.models: Dict[str, YOLO] = {}

    def load_model(self, model_type: str) -> YOLO:

        if model_type not in MODEL_PATHS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported models: {list(MODEL_PATHS.keys())}"
            )

        # Return already-loaded model
        if model_type in self.models:
            return self.models[model_type]

        model_path = MODEL_PATHS[model_type]

        if not model_path.exists():
            fallback_fls = MODELS_DIR / "fls" / "best.pt"
            fallback_root = BASE_DIR / "yolov8s.pt"
            if fallback_fls.exists():
                model_path = fallback_fls
            elif fallback_root.exists():
                model_path = fallback_root
            else:
                raise FileNotFoundError(
                    f"{model_type.upper()} model weights not found: {model_path}"
                )

        print(f"[ML] Loading {model_type.upper()} model...")
        print(f"[ML] Weights: {model_path}")

        model = YOLO(str(model_path))

        self.models[model_type] = model

        print(f"[ML] {model_type.upper()} model loaded successfully")
        print(f"[ML] Classes: {model.names}")

        return model

    def get_available_models(self):
        return {
            name: {
                "available": path.exists(),
                "path": str(path),
            }
            for name, path in MODEL_PATHS.items()
        }


model_manager = ModelManager()