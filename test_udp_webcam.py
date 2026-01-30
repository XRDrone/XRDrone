import cv2

import subprocess

from ultralytics import YOLO

import socket

import json

import time
 
RTSP_URL = "rtsp://127.0.0.1:8554/stream"
  
#FPS Measurement Setup
t0 = time.time()
count = 0

# ---- Open webcam (built-in camera) ----
cap = cv2.VideoCapture(0)  # try 0 first; 1,2,... if needed
if not cap.isOpened():
    print("Could not open webcam")
    exit()
 
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
 
print(f"Capture: {width}x{height} @ {fps} fps")
 
# ---- Load YOLO model ----

model = YOLO("yolov8n.pt")  # or your yolov11 model path
 
# ---- FFmpeg subprocess for RTSP streaming ----

ffmpeg = subprocess.Popen([

    "ffmpeg", "-re",

    "-f", "rawvideo", "-pix_fmt", "bgr24",

    "-s", f"{width}x{height}", "-r", str(fps),

    "-i", "-",

    "-an", "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",

    "-pix_fmt", "yuv420p",

    "-f", "rtsp", RTSP_URL

], stdin=subprocess.PIPE)
 
# ---- UDP socket for metadata to Unity ----

UDP_IP = "127.0.0.1"

UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
 
frame_id = 0
 
print("Streaming started. Press 'q' in preview window to quit.")
 
# ---- Main loop ----

while True:

    ret, frame = cap.read()

    if not ret:

        print("Frame grab failed, stopping.")

        break
 
    # Run YOLO detection (only people, class 0)

    results = model(frame, classes=[0], verbose=False)

    r = results[0]
 
    # Plot bounding boxes for RTSP preview

    frame_annot = r.plot()
 
    # ---- Build detection metadata for this frame ----

    detections = []

    boxes = r.boxes
 
    # boxes.xywhn: normalized [cx, cy, w, h] in [0,1]

    if boxes is not None and len(boxes) > 0:

        xywhn = boxes.xywhn.cpu().numpy()

        confs = boxes.conf.cpu().numpy()

        clses = boxes.cls.cpu().numpy()
 
        for i, (cx, cy, w, h) in enumerate(xywhn):

            det = {

                "id": int(i),             # simple per-frame ID (can replace with tracker ID later)

                "cls": int(clses[i]),     # should be 0 for "person"

                "conf": float(confs[i]),

                "cx": float(cx),          # normalized 0–1

                "cy": float(cy),          # normalized 0–1

                "w": float(w),

                "h": float(h)

            }

            detections.append(det)
 
    meta = {

        "frame_id": frame_id,

        "timestamp": time.time(),

        "width": width,

        "height": height,

        "detections": detections

    }
 
    # Send JSON metadata via UDP

    try:

        msg = json.dumps(meta).encode("utf-8")

        sock.sendto(msg, (UDP_IP, UDP_PORT))

    except Exception as e:

        print("UDP send error:", e)
 
    frame_id += 1
 
    # Show preview

    cv2.imshow("Preview", frame_annot)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
 
    # Send annotated frame to FFmpeg

    try:

        ffmpeg.stdin.write(frame_annot.tobytes())

    except BrokenPipeError:

        print("FFmpeg pipe closed.")

        break
     
    # FPS Calculation
    count += 1
    if time.time() - t0 >= 1.0:
        print(f"[MEASURE] actual_loop_fps={count/(time.time()-t0):.2f}")
        t0 = time.time()
        count = 0
 
# ---- Cleanup ----

cap.release()

cv2.destroyAllWindows()

ffmpeg.stdin.close()

ffmpeg.wait()

sock.close()

print("Stopped.")


 
