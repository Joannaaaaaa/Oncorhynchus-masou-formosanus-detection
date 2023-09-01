from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("data/Yolov8n.yaml")
    model.train(data="data/fish.yaml",
                mode="detect",
                epochs=3,
                imgsz=640,
                batch=2,
                lr0=0.0001,
                amp=False,
                device=0)
