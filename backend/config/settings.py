"""
Traffic Detection Backend Configuration
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
YOLO_MODEL_PATH = MODELS_DIR / "best.pt"
SSD_MODEL_PATH = MODELS_DIR / "ssd300_vgg16_coco.pth"

# Ensure models directory exists
MODELS_DIR.mkdir(exist_ok=True)

# Set torch hub directory to models folder
os.environ['TORCH_HOME'] = str(MODELS_DIR)

# Detection Settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.1
MAX_CONFIDENCE = 1.0

# Class Configuration
CLASS_NAMES = [
    "Car", "Pedestrian", "Van", "Cyclist",
    "Truck", "Misc", "Tram", "Person_sitting"
]

# BGR colors for OpenCV
CLASS_COLORS = {
    "Car": (255, 0, 0),
    "Pedestrian": (0, 255, 0),
    "Van": (0, 0, 255),
    "Cyclist": (255, 255, 0),
    "Truck": (255, 0, 255),
    "Misc": (0, 255, 255),
    "Tram": (128, 0, 128),
    "Person_sitting": (255, 165, 0)
}

# SSD COCO class mapping (relevant traffic classes)
SSD_TRAFFIC_CLASSES = {
    1: "Pedestrian",   # person
    2: "Cyclist",      # bicycle
    3: "Car",          # car
    4: "Cyclist",      # motorcycle
    6: "Truck",        # bus
    7: "Car",          # train -> map to car
    8: "Truck",        # truck
}

# API Settings
API_PREFIX = "/api"
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# Video Processing
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]

# Camera Settings
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_FPS = 30
