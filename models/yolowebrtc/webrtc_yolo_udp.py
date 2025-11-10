# whep_yolo_udp.py
# Subscribe to a WHEP endpoint → receive video (aiortc) → YOLO → UDP JSON to Unity
# Keeps a local HUD preview using your hud.draw_hud()

import argparse, asyncio, json, time, cv2, socket, httpx
from collections import deque
from ultralytics import YOLO
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole
from hud import draw_hud  # from your hud.py

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser("WHEP → YOLO → UDP (Unity)")
    p.add_argument("--whep", required=True, help="WHEP URL, e.g. http://localhost:8889/stream/whep")
    p.add_argument("--model", default="yolo12n.pt")
    p.add_argument("--imgsz", type=int, default=520)
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--classes", default="0", help="comma ids, e.g. '0' for person")
    p.add_argument("--udp_ip", default="127.0.0.1")
    p.add_argument("--udp_port", type=int, default=9999)
    p.add_argument("--no-show", dest="show", action="store_false")
    p.add_argument("--show", dest="show", action="store_true", default=True)
    return p.parse_args()

# ---------- Latest frame buffer ----------
class LatestFrame:
    def __init__(self):
        self.frame = None  # numpy BGR
        self.ts = 0.0
latest = LatestFrame()

# ---------- Video sink (captures incoming frames) ----------
class VideoSink(MediaStreamTrack):
    kind = "video"
    def __init__(self, track):
        super().__init__()
        self.track = track
    async def recv(self):
        frame = await self.track.recv()
        img = frame.to_ndarray(format="bgr24")
        latest.frame = img
        latest.ts = time.time()
        return frame

# ---------- UDP emitter ----------
class UdpEmitter:
    def __init__(self, ip, port):
        self.dest = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(self, res, w, h, names):
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xys = res.boxes.xyxy.cpu().numpy()
            clz = res.boxes.cls.cpu().numpy()
            con = res.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), c, p in zip(xys, clz, con):
                cid = int(c); conf = float(p)
                dets.append({
                    "cls": names.get(cid, str(cid)),
                    "id": cid, "conf": conf,
                    "x": float(x1), "y": float(y1),
                    "w": float(x2-x1), "h": float(y2-y1)
                })
        pkt = {"ts": time.time(), "w": int(w), "h": int(h), "detections": dets}
        self.sock.sendto(json.dumps(pkt, separators=(",", ":")).encode("utf-8"), self.dest)

# ---------- Create WHEP peer (POST offer, set answer) ----------
async def create_whep_peer(whep_url: str) -> RTCPeerConnection:
    pc = RTCPeerConnection()
    blackhole = MediaBlackhole()

    @pc.on("track")
    def on_track(track):
        if track.kind == "video":
            pc.addTrack(VideoSink(track))
        else:
            blackhole.addTrack(track)

    # Create an offer with RecvOnly transceivers (subscribe)
    pc.addTransceiver("video", direction="recvonly")
    pc.addTransceiver("audio", direction="recvonly")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # POST SDP offer body (content-type application/sdp) like your Unity script does
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.post(whep_url, content=pc.localDescription.sdp,
                              headers={"Content-Type": "application/sdp"})
        r.raise_for_status()
        answer_sdp = r.text

    # Set the remote answer
    answer = RTCSessionDescription(sdp=answer_sdp, type="answer")
    await pc.setRemoteDescription(answer)
    return pc

# ---------- YOLO + HUD loop ----------
def run_yolo_loop(args, model: YOLO, udp: UdpEmitter):
    names = getattr(getattr(model, "model", None), "names", {}) or {}
    classes = [int(s) for s in args.classes.split(",") if s.strip() != ""]
    fps_hist, inf_hist = deque(maxlen=30), deque(maxlen=30)
    t_prev = time.time()

    while True:
        if latest.frame is None:
            time.sleep(0.01)
            continue

        now = time.time()
        dt = now - t_prev
        t_prev = now
        if dt > 0:
            fps_hist.append(1.0/dt)

        res = model.predict(
            latest.frame,
            stream=False,
            conf=args.conf,
            imgsz=args.imgsz,
            classes=classes,
            verbose=False
        )[0]

        inf_ms = float(res.speed.get("inference", 0.0))
        inf_hist.append(inf_ms)

        H, W = res.orig_shape[:2]
        udp.send(res, W, H, names)  # → Unity

        if args.show:
            frame = res.plot()
            avg_fps = sum(fps_hist)/max(1,len(fps_hist))
            avg_inf = sum(inf_hist)/max(1,len(inf_hist))
            people = int((res.boxes.cls == 0).sum()) if res.boxes is not None else 0

            lines = [
                f"FPS: {avg_fps:5.2f}",
                f"Model latency: {avg_inf:5.1f} ms",
                f"People: {people}"
            ]
            frame = draw_hud(frame, lines, anchor="tl")  # uses your hud.py
            vis_w = 1280
            scale = vis_w / frame.shape[1]
            frame = cv2.resize(frame, (vis_w, int(frame.shape[0]*scale)))
            cv2.imshow("WHEP Live (debug)", frame)
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()

# ---------- Entrypoint ----------
async def main():
    args = parse_args()
    pc = await create_whep_peer(args.whep)
    try:
        model = YOLO(args.model)
        udp = UdpEmitter(args.udp_ip, args.udp_port)
        await asyncio.to_thread(run_yolo_loop, args, model, udp)
    finally:
        await pc.close()

if __name__ == "__main__":
    import time  # used above
    asyncio.run(main())
