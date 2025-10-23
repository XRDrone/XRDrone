from ultralytics import YOLO

model = YOLO('yolo12n-seg.pt') 

results = model(source=0, show=True, conf=0.4, save=False)
