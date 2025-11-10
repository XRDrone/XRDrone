STEPS TO RUN LIVE OBJECT DETECTION:
1. You will need to create a python virtual environment and install the necessary packages before running the program:

python3 -m venv yolovenv OR python -m venv yolovenv

source yolovenv/bin/activate # for mac
.\yolovenv\Scripts\activate # for windows

pip install --upgrade pip OR python -m pip install --upgrade pip
pip install ultralytics
pip install opencv-python

2. Please run the following command in the terminal. When object detection is running, you can press the ESC key to exit out of the program.

python main.py

Output of the video will be saved in the same directory.

3. Once the program finishes, to exit out of the virutal environment, type "deactivate" in the terminal.
