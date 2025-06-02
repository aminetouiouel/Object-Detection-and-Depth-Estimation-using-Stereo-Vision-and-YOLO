import cv2
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

def detect_objects(image):
    results = model(image)
    detections = results[0].boxes.data.cpu().numpy()
    return detections

def draw_detections(img, detections):
    for detection in detections:
        x1, y1, x2, y2, conf, cls_id = detection
        cls_name = model.names[int(cls_id)]
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, cls_name, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)
    return img
