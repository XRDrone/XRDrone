from ultralytics import YOLO

# Load a COCO-pretrained YOLO12n model
model = YOLO("yolo12n.pt")

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=1, imgsz=640)

# Run inference with the YOLO12n model on the 'bus.jpg' image
results = model("/Users/troy/Desktop/XRDrone/models/videos/maddy.MOV", save=True, project=".")