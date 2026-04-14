from ultralytics import YOLO
import cv2
import os

# Load trained model
MODEL_PATH = r"C:\Users\Sagar\Desktop\Data\onion_yolo_project\scripts\runs\detect\onion_detector8\weights\best.pt"

model = YOLO(MODEL_PATH)


# Input image path
IMAGE_PATH = "res3.jpg"


results = model.predict(source=IMAGE_PATH, conf=0.25, save=True)


# Count onions
boxes = results[0].boxes
count = len(boxes)

print("Onion Count:", count)


# Show image 
img = results[0].plot()

cv2.imshow("Onion Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Save custom output
output_dir = "outputs/images"
os.makedirs(output_dir, exist_ok=True)

save_path = os.path.join(output_dir, "result2.jpg")
cv2.imwrite(save_path, img)

print("Saved result at:", save_path)