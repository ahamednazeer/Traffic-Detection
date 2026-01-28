"""
SSD (Single Shot Detector) Implementation
Uses torchvision's SSD300 with VGG16 backbone
"""
import os
import numpy as np
from typing import List
from pathlib import Path
import torch
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

from .base_detector import BaseDetector, Detection
from config.settings import SSD_TRAFFIC_CLASSES, CLASS_COLORS, MODELS_DIR
from utils.download import set_download_state, reset_download_state


class SSDDetector(BaseDetector):
    """SSD300 object detector using torchvision."""
    
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = None
        
        # Set torch hub to models directory
        torch.hub.set_dir(str(MODELS_DIR))
    
    def load_model(self) -> bool:
        """Load the SSD model with COCO pre-trained weights."""
        try:
            # Track download state
            set_download_state(
                is_downloading=True,
                model_name="SSD300 (VGG16)",
                progress=0
            )
            
            # Load pre-trained SSD300 with VGG16 backbone
            weights = SSD300_VGG16_Weights.COCO_V1
            self.model = ssd300_vgg16(weights=weights)
            self.model.to(self.device)
            self.model.eval()
            
            # Get transforms
            self.transform = weights.transforms()
            
            self.is_loaded = True
            reset_download_state()
            print(f"SSD300 model loaded (COCO weights, device: {self.device})")
            print(f"Model cache: {MODELS_DIR}")
            return True
        except Exception as e:
            reset_download_state()
            print(f"Failed to load SSD model: {e}")
            self.is_loaded = False
            return False
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform SSD detection on an image.
        
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
            # Convert BGR to RGB
            image_rgb = image[:, :, ::-1].copy()
            
            # Convert to PIL Image then to tensor
            from PIL import Image
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            input_tensor = self.transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                predictions = self.model(input_batch)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred["boxes"].cpu().numpy()
            scores = pred["scores"].cpu().numpy()
            labels = pred["labels"].cpu().numpy()
            
            # Get image dimensions for denormalization
            h, w = image.shape[:2]
            
            for box, score, label in zip(boxes, scores, labels):
                if score >= confidence_threshold:
                    # Map COCO class to traffic class
                    label_int = int(label)
                    if label_int in SSD_TRAFFIC_CLASSES:
                        class_name = SSD_TRAFFIC_CLASSES[label_int]
                        
                        # Get coordinates (already in pixel format from SSD)
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Clamp to image bounds
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=float(score),
                            bbox=(x1, y1, x2, y2),
                            class_id=label_int
                        ))
        
        except Exception as e:
            print(f"SSD detection error: {e}")
            import traceback
            traceback.print_exc()
        
        return detections
    
    def get_model_name(self) -> str:
        return "SSD300 (VGG16)"
