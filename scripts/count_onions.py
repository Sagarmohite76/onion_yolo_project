from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("runs/detect/onion_detector8/weights/best.pt")

# Load image
img = cv2.imread("res4.jpg")

# Predict
results = model(img)

# Count onions
count = len(results[0].boxes)

print("Onion Count:", count)

# Show image
annotated = results[0].plot()
cv2.imshow("Detection", annotated)
cv2.waitKey(0)