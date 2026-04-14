# count_onions.py
# Script to count onions in images or videos
from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/onion_detector/weights/best.pt")

# Load image
img = cv2.imread("test.jpg")

# Predict
results = model(img)

# Count onions
count = len(results[0].boxes)

print("Onion Count:", count)

# Show image
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)