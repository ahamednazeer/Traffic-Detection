"""
YOLO v11 Detector Implementation
"""
import numpy as np
from typing import List
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection
from config.settings import CLASS_NAMES, YOLO_MODEL_PATH
from utils.download import set_download_state, reset_download_state


class YOLODetector(BaseDetector):
    """YOLO v11 object detector using ultralytics."""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path or str(YOLO_MODEL_PATH))
    
    def load_model(self) -> bool:
        """Load the YOLO model."""
        try:
            set_download_state(
                is_downloading=True,
                model_name="YOLO v11",
                progress=0
            )
            
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            
            reset_download_state()
            print(f"YOLO model loaded from {self.model_path}")
            return True
        except Exception as e:
            reset_download_state()
            print(f"Failed to load YOLO model: {e}")
            self.is_loaded = False
            return False
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform YOLO detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format from OpenCV)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of Detection objects
        """
        if not self.is_loaded:
            if not self.load_model():
                return []
        
        detections = []
        
        try:
            # Run inference
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        if class_id < len(CLASS_NAMES):
                            class_name = CLASS_NAMES[class_id]
                        else:
                            class_name = f"Class_{class_id}"
                        
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            class_id=class_id
                        ))
        
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def get_model_name(self) -> str:
        return "YOLO v11"
