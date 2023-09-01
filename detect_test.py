from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")
result = model.predict(
    source="000000.jpg",
    mode="predict",
    save=True,
    device=0
)
