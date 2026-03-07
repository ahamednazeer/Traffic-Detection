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
TRAFFIC_SIGN_MODEL_PATH = MODELS_DIR / "traffic_sign.pt"
ACCIDENT_MODEL_PATH = MODELS_DIR / "accident_train" / "weights" / "best.pt"
ACCIDENT_MODEL_URL = ""


def _load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file."""
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# Load backend-local env first, then project-root fallback.
_load_env_file(BASE_DIR / ".env")
_load_env_file(BASE_DIR.parent / ".env")

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
    "Truck", "Misc", "Tram", "Person_sitting",
    "Traffic Light", "Red Light", "Yellow Light", "Green Light",
    "Stop Sign", "Speed Limit", "Yield", "Warning", "Traffic Sign",
    "Speed Limit 10", "Speed Limit 20", "Speed Limit 30", "Speed Limit 40",
    "Speed Limit 50", "Speed Limit 60", "Speed Limit 70", "Speed Limit 80",
    "Speed Limit 90", "Speed Limit 100", "Speed Limit 110", "Speed Limit 120"
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
    "Person_sitting": (255, 165, 0),
    "Bus": (0, 128, 255),  # Orange-ish for buses
    "Traffic Light": (0, 255, 255),  # Yellow/Cyan mix for high visibility
    "Red Light": (0, 0, 255),        # Red (BGR)
    "Yellow Light": (0, 255, 255),   # Yellow (BGR)
    "Green Light": (0, 255, 0),      # Green (BGR)
    "Stop Sign": (0, 0, 255),        # Red
    "Speed Limit": (255, 255, 255),  # White
    "Yield": (0, 255, 255),          # Yellow
    "Warning": (0, 165, 255),        # Orange
    "Prohibitory": (0, 0, 255),      # Red
    "Prohibitory": (0, 0, 255),      # Red
    "Mandatory": (255, 0, 0),        # Blue
    "Traffic Sign": (0, 255, 255),   # Yellow
    
    # Speed Limits (White/Red)
    "Speed Limit 10": (255, 255, 255),
    "Speed Limit 20": (255, 255, 255),
    "Speed Limit 30": (255, 255, 255),
    "Speed Limit 40": (255, 255, 255),
    "Speed Limit 50": (255, 255, 255),
    "Speed Limit 60": (255, 255, 255),
    "Speed Limit 70": (255, 255, 255),
    "Speed Limit 80": (255, 255, 255),
    "Speed Limit 90": (255, 255, 255),
    "Speed Limit 100": (255, 255, 255),
    "Speed Limit 110": (255, 255, 255),
    "Speed Limit 120": (255, 255, 255),
    "Accident": (0, 0, 255),
    "accident": (0, 0, 255),
    "Vehicle": (255, 0, 0),
    "vehicle": (255, 0, 0),
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

# Email Notification Settings
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", os.getenv("SMTP_USER", "")).strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
# Gmail app passwords are shown grouped with spaces; normalize for SMTP auth.
if "gmail.com" in SMTP_HOST.lower():
    SMTP_PASSWORD = SMTP_PASSWORD.replace(" ", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USERNAME)
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "Traffic Detection System")
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
ALERT_EMAIL_ENABLED = os.getenv("ALERT_EMAIL_ENABLED", "true").lower() == "true"
ALERT_RECIPIENT_EMAIL = os.getenv(
    "ALERT_RECIPIENT_EMAIL",
    os.getenv("NOTIFY_EMAIL", os.getenv("USER_EMAIL", ""))
).strip()

# Video Processing
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]
SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".webp"]

# Camera Settings
CAMERA_FRAME_WIDTH = 640
CAMERA_FRAME_HEIGHT = 480
CAMERA_FPS = 30
