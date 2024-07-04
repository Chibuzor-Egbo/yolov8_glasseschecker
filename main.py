import logging
from ultralytics import YOLO

# Suppress logging messages
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)

# Model
model = YOLO("best.pt")


results = model.predict("passport.jpg", imgsz=320, conf=0.7)

# Check for detections
if any(len(result) > 0 for result in results):
    print(True)
else:
    print(False)