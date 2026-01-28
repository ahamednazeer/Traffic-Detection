"""
Base Detector Abstract Class
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
import numpy as np


class Detection:
    """Represents a single detection result."""
    
    def __init__(
        self,
        class_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],  # x1, y1, x2, y2
        class_id: int = -1
    ):
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox
        self.class_id = class_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class": self.class_name,
            "confidence": float(self.confidence),
            "bbox": {
                "x1": int(self.bbox[0]),
                "y1": int(self.bbox[1]),
                "x2": int(self.bbox[2]),
                "y2": int(self.bbox[3])
            },
            "class_id": self.class_id
        }


class BaseDetector(ABC):
    """Abstract base class for object detectors."""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load the model. Returns True if successful."""
        pass
    
    @abstractmethod
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of Detection objects
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the detector."""
        pass
    
    def get_class_names(self) -> List[str]:
        """Return list of class names this detector can identify."""
        from config.settings import CLASS_NAMES
        return CLASS_NAMES
    
    def get_class_colors(self) -> Dict[str, Tuple[int, int, int]]:
        """Return color mapping for classes."""
        from config.settings import CLASS_COLORS
        return CLASS_COLORS
    
    def unload_model(self):
        """Unload the model to free memory."""
        self.model = None
        self.is_loaded = False
