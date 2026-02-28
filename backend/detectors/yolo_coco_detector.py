"""
YOLO COCO Detector Implementation - Pre-trained on COCO dataset
Detects 80 classes including various vehicle types
"""
import numpy as np
from typing import List
import cv2
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection
from utils.download import set_download_state, reset_download_state


# COCO classes relevant to traffic detection (mapped to standard names)
COCO_TRAFFIC_MAPPING = {
    0: "Pedestrian",      # person
    1: "Cyclist",         # bicycle
    2: "Car",             # car
    3: "Cyclist",         # motorcycle
    5: "Bus",             # bus
    6: "Car",             # train -> Car
    7: "Truck",           # truck
    9: "Traffic Light",   # traffic light
    11: "Stop Sign",      # stop sign
}

# All 80 COCO class names for reference
COCO_CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class YOLOCocoDetector(BaseDetector):
    """YOLO detector using pre-trained COCO weights for broader detection."""
    
    def __init__(self, model_size: str = "n"):
        """
        Initialize YOLO COCO detector.
        
        Args:
            model_size: Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (xlarge)
        """
        # Use ultralytics pretrained model (will download automatically)
        super().__init__(f"yolo11{model_size}.pt")
        self.model_size = model_size
        self.filter_traffic = True  # Only return traffic-related detections
    
    def load_model(self) -> bool:
        """Load the YOLO COCO model."""
        try:
            set_download_state(
                is_downloading=True,
                model_name=f"YOLO11-{self.model_size.upper()} (COCO)",
                progress=0
            )
            
            # This will auto-download pretrained COCO weights
            self.model = YOLO(self.model_path)
            self.is_loaded = True
            
            reset_download_state()
            print(f"YOLO COCO model loaded: {self.model_path}")
            return True
        except Exception as e:
            reset_download_state()
            print(f"Failed to load YOLO COCO model: {e}")
            self.is_loaded = False
            return False
    
    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        """
        Perform detection using COCO-trained YOLO.
        
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
            results = self.model(
                image,
                conf=confidence_threshold,
                verbose=False,
                save=False,
                save_txt=False,
                save_conf=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Filter to traffic-related classes only
                        if self.filter_traffic and class_id not in COCO_TRAFFIC_MAPPING:
                            continue
                        
                        # Get class name (use mapped name or original COCO name)
                        if class_id in COCO_TRAFFIC_MAPPING:
                            class_name = COCO_TRAFFIC_MAPPING[class_id]
                        else:
                            class_name = COCO_CLASS_NAMES[class_id] if class_id < len(COCO_CLASS_NAMES) else f"Class_{class_id}"
                        
                        # Determine Traffic Light State
                        if class_name == "Traffic Light":
                            class_name = self._determine_signal_state(image, (x1, y1, x2, y2))
                        
                        detections.append(Detection(
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            class_id=class_id
                        ))
        
        except Exception as e:
            print(f"YOLO COCO detection error: {e}")
        
        return detections
    
    def _determine_signal_state(self, image: np.ndarray, bbox: tuple) -> str:
        """
        Analyze the bounding box to determine if the traffic light 
        is Red, Yellow, or Green.
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return "Traffic Light"
            
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges
        # Red has two ranges in HSV (0-10 and 160-180)
        # Relaxed thresholds: Saturation 30, Value 30 to detect washed out/dark lights
        lower_red1 = np.array([0, 30, 30])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 30, 30])
        upper_red2 = np.array([180, 255, 255])
        
        lower_yellow = np.array([15, 30, 30])
        upper_yellow = np.array([35, 255, 255])
        
        lower_green = np.array([40, 30, 30]) 
        upper_green = np.array([90, 255, 255])
        
        # Create masks
        mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)
        
        mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
        mask_green = cv2.inRange(hsv_roi, lower_green, upper_green)
        
        # Count pixels
        red_pixels = cv2.countNonZero(mask_red)
        yellow_pixels = cv2.countNonZero(mask_yellow)
        green_pixels = cv2.countNonZero(mask_green)
        
        # Threshold to consider it valid (e.g. at least 0.1% of pixels or absolute count)
        total_pixels = roi.shape[0] * roi.shape[1]
        threshold = max(5, total_pixels * 0.001) # Very low threshold for video/small objects
        
        states = {
            "Red Light": red_pixels,
            "Yellow Light": yellow_pixels,
            "Green Light": green_pixels
        }
        
        # Get the color with max pixels
        best_state = max(states, key=states.get)
        max_pixels = states[best_state]
        
        if max_pixels > threshold:
            return best_state
            
        return "Traffic Light"
    
    def get_model_name(self) -> str:
        return f"YOLO11-{self.model_size.upper()} (COCO)"
