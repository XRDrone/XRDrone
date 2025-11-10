import cv2, subprocess
from ultralytics import YOLO
 
RTSP_URL = "rtsp://127.0.0.1:8554/stream"
 
# Use webcam instead of MP4
cap = cv2.VideoCapture(0)  # change 0 to 1 if you have multiple webcams
if not cap.isOpened():
    print("Could not open webcam")
    exit()
 
width  = int(cap.get(3))
height = int(cap.get(4))
fps    = cap.get(5) or 30
 
model = YOLO("yolov8n.pt")
 
ffmpeg = subprocess.Popen([
    "ffmpeg", "-re",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(int(fps)),
    "-i", "-",
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-f", "rtsp", RTSP_URL
], stdin=subprocess.PIPE)
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    results = model(frame, classes=[0])  # Detect only people
    frame = results[0].plot()
 
    try:
        ffmpeg.stdin.write(frame.tobytes())
    except:
        break