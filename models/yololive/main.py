from ultralytics import YOLO
import cv2, time
from collections import deque
from hud import draw_hud

model = YOLO("yolo12n.pt")
fps_hist = deque(maxlen=30)
inf_hist = deque(maxlen=30)
t_prev = time.time()

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 24, (1280, 720))

for r in model(source=0, 
               stream=True, 
               show=False, 
               classes=[0], 
               conf=0.4, 
               imgsz=520):

    now = time.time(); dt = now - t_prev; t_prev = now
    if dt > 0: fps_hist.append(1/dt)
    inf_hist.append(r.speed['inference'])

    frame = r.plot()

    avg_fps = sum(fps_hist)/len(fps_hist)
    avg_inf = sum(inf_hist)/len(inf_hist)
    people = int((r.boxes.cls == 0).sum())

    lines = [
        f"FPS: {avg_fps:5.2f}",
        f"Model latency: {avg_inf:5.1f} ms",
        f"People: {people}"
    ]

    frame = draw_hud(frame, lines, anchor="tl")
    frame = cv2.resize(frame, (1280, 720))

    out.write(frame)
    cv2.imshow("Live", frame)
    if cv2.waitKey(1) == 27: break

out.release()

cv2.destroyAllWindows()
