import cv2, subprocess
from ultralytics import YOLO
 
VIDEO_IN = "people.mp4"
RTSP_URL = "rtsp://127.0.0.1:8554/stream"
 
cap = cv2.VideoCapture(VIDEO_IN)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(5) or 30
 
model = YOLO("yolov8n.pt")
 
ffmpeg = subprocess.Popen([
    "ffmpeg", "-re",
    "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", str(int(fps)),
    "-i", "-",
    "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
    "-pix_fmt", "yuv420p",
    "-f", "rtsp", RTSP_URL
], stdin=subprocess.PIPE)
 
while True:
    ret, frame = cap.read()
    if not ret:                         # means video ended
        cap.release()                   # close file
        cap = cv2.VideoCapture(VIDEO_IN)  # reopen it (loop)
        continue
 
    results = model(frame, classes=[0], verbose=False)  # detect only people
    frame = results[0].plot()                           # draw boxes
 
    try:
        ffmpeg.stdin.write(frame.tobytes())
    except:
        break