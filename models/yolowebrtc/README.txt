python3 -m venv yolovenv

source yolovenv/bin/activate # for mac
.\yolovenv\Scripts\activate # for windows

pip install --upgrade pip
pip install ultralytics
pip install opencv-python
pip install aiortc 
pip install av 
pip install httpx

python whep_yolo_udp.py --whep http://localhost:8889/stream/whep --model yolo12n.pt --classes 0 --imgsz 520 --conf 0.4