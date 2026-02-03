"""
Traffic Sign Detector Implementation
"""
import os
import requests
import numpy as np
from typing import List
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection
from config.settings import TRAFFIC_SIGN_MODEL_PATH
from utils.download import set_download_state, reset_download_state


class TrafficSignDetector(BaseDetector):
    """
    YOLO-based detector specialized for Traffic Signs.
    Auto-downloads weights if not present.
    """
    
    DOWNLOAD_URL = "https://github.com/muhammadrizwan11/Traffic-Sign-Detection-Using-Yolov8/raw/main/best.pt"
    
    def __init__(self):
        super().__init__(str(TRAFFIC_SIGN_MODEL_PATH))
        # No hardcoded mapping, rely on model.names

    def load_model(self) -> bool:
        """Load the model, downloading if necessary."""
        try:
            # Check if model exists
            if not os.path.exists(self.model_path):
                print(f"Traffic sign model not found at {self.model_path}. Downloading...")
                set_download_state(
                    is_downloading=True,
                    model_name="Traffic Sign Model (YOLOv8)",
                    progress=0
                )
                
                try:
                    response = requests.get(self.DOWNLOAD_URL, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    block_size = 1024 # 1 Kibibyte
                    wrote = 0
                    
                    with open(self.model_path, 'wb') as f:
                        for data in response.iter_content(block_size):
                            wrote += len(data)
                            f.write(data)
                    
                    print(f"Downloaded traffic sign model to {self.model_path}")
                    
                except Exception as e:
                    print(f"Failed to download model: {e}")
                    # Clean up partial file
                    if os.path.exists(self.model_path):
                        os.remove(self.model_path)
                    reset_download_state()
                    return False

            set_download_state(
                is_downloading=True,
                model_name="Loading Traffic Sign Model...",
                progress=100
            )

            self.model = YOLO(self.model_path)
            self.is_loaded = True
            
            # Log available classes for debugging
            if hasattr(self.model, 'names'):
                print(f"Traffic Sign Classes: {self.model.names}")
            
            reset_download_state()
            return True
            
        except Exception as e:
            reset_download_state()
            print(f"Failed to load Traffic Sign model: {e}")
            self.is_loaded = False
            return False
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform detection.
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        detections = []
        
        try:
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Use model's internal name
                        class_name = result.names[class_id] if class_id in result.names else f"Sign_{class_id}"

                        detections.append(Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            class_id=class_id
                        ))
        
        except Exception as e:
            print(f"Traffic sign detection error: {e}")
        
        return detections
    
    def get_model_name(self) -> str:
        return "Traffic Sign (Limited Classes)"

    def get_class_colors(self):
        """Custom colors for this specific model's classes."""
        return {
            "Stop": (0, 0, 255),          # Red
            "Red Light": (0, 0, 255),     # Red
            "Green Light": (0, 255, 0),   # Green
            "Speed Limit 10": (255, 255, 255), # White
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
        }
