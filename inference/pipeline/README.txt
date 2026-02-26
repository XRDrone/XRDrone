FILES:
main.py - The central script that runs the full detection pipeline.
hud.py - A utility module for drawing visual elements onto video frames.
merger.py - Takes raw YOLO results from multiple models and produces a clean, merged list of detections.
output_formatter.py (SOON) - This module takes the merged detections from merger.py and converts them into a format that is easy for sending to Unity.

EXAMPLE MERGER OUTPUT:
[
  {
    'bbox_xyxy': [170.86, 131.59, 1600.36, 1074.59],
    'class': 'person',
    'confidence': 0.9262641072273254,
    'mask': None,
    'source': 'people',
    'timestamp': 1763754395.278698
  },
  {
    'bbox_xyxy': [424.53, 14.09, 1897.86, 1051.13],
    'class': 'smoke',
    'confidence': 0.4812464118003845,
    'mask': None,
    'source': 'fire',
    'timestamp': 1763754395.278698
  }
]

STEPS TO RUN LIVE OBJECT DETECTION AND INSTANCE SEGMENTATION:
1. You will need to create a python virtual environment and install the necessary packages before running the program:
2. Run the main program: main.py 
3. Once the program finishes, to exit out of the virutal environment, press escape.

COMMANDS (RUN IN ORDER):
python3 -m venv yolovenv

source yolovenv/bin/activate # mac/linux
.\yolovenv\Scripts\activate # windows

pip install --upgrade pip 
pip install ultralytics
pip install opencv-python

python3 main.py

SETTINGS:  
O - Toggle bounding boxes on/off
P - Toggle segmentation mode on/off
K - Toggle the people model on/off
L - Toggle the fire/smoke model on/off
R - Toggle livestream recording on/off
ESC - Quit the program / close the live window

deactivate # exiting from virtual environment
