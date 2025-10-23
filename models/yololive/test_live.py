from ultralytics import YOLO

model = YOLO('yolov10n.pt') 

results = model(source=0, show=True, conf=0.4, save=False)
