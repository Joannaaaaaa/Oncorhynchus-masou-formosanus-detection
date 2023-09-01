from ultralytics import YOLO


model = YOLO("runs/detect/train/weights/best.pt")  # training weight
result = model.predict(
    source="000000.jpg",  # test image
    mode="predict",
    save=True,
    device=0
)
