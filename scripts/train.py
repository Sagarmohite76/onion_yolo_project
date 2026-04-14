# train.py
# Script to train the YOLO model
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")

# Train
model.train(
    data="C:\\Users\\Sagar\\Desktop\\Data\\onion_yolo_project\\dataset\\data.yaml",
    epochs=50,
    imgsz=512,
    batch=16,
    name="onion_detector"
)