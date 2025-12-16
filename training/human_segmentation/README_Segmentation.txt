1) In your directory you should have:

aeroscapes-DatasetNinja
prepare_aeroscapes_person.py
yolo11n-seg.pt (or some other pretrained segmentation model)

2) Install dependencies

pip install numpy pillow opencv-python

3) Clean the dataset

python .\prepare_aeroscapes_person.py

4) Now run your training script

python .\train_aeroscapes_yolo11-seg_960.py

5) Follow progress

python -m tensorboard.main --logdir "E:\XRDrone\runs" --port 6008
