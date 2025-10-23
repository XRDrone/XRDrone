python3 -m venv yolovenv

source yolovenv/bin/activate # for mac
.\yolovenv\Scripts\activate # for windows

pip install --upgrade pip
pip install ultralytics
pip install opencv-python

python3 test_live.py