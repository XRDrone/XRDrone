WARNING: Training can be disk-space intensive (especially with cache=disk). Make sure you have at least 500 GB of free space available before starting.

1) In your directory you should have:

WiSARDv1
yolo11n.pt (or some pretrained model)

2) Run the script to clean for only VIS folders:

python prepare_wisard_vis.py

3) Install libraries:

python -m pip install -U ultralytics torch

4) Now run the training script:

python train_wisard_yolo11_960.py

5) Watch your models updates:

python -m tensorboard.main --logdir "E:\XRDrone\runs" --port 6008

