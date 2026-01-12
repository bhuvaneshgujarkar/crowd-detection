import torch
from ultralytics import YOLO

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# YOLO test
model = YOLO("yolov8n.pt")
print("YOLO model loaded successfully")