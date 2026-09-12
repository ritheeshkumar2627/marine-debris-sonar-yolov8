import numpy as np
from ultralytics import YOLO

model = YOLO("models/fls/best.pt")

print("Classes in YOLO model:", model.names)

# Create a sample synthetic grayscale/RGB sonar-like image (640x640)
sample_img = np.zeros((640, 640, 3), dtype=np.uint8)

# Run model inference
results = model(sample_img, conf=0.10)

for result in results:
    print("Boxes detected:", len(result.boxes))
    print("Model inference successful!")