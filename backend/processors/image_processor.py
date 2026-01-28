"""
Image Processing Utilities
"""
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from detectors.base_detector import BaseDetector, Detection
from config.settings import CLASS_COLORS


class ImageProcessor:
    """Handles image processing and visualization."""
    
    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Detection],
        line_thickness: int = 2,
        font_scale: float = 0.6
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on image.
        
        Args:
            image: Input image (BGR format)
            detections: List of Detection objects
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of text labels
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = CLASS_COLORS.get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)
            
            # Create label
            label = f"{det.class_name}: {det.confidence:.2f}"
            
            # Get label size
            (label_width, label_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            
            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - label_height - 10),
                (x1 + label_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2
            )
        
        return annotated
    
    @staticmethod
    def process_image(
        image: np.ndarray,
        detector: BaseDetector,
        confidence_threshold: float = 0.5,
        draw_boxes: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process an image with detection.
        
        Args:
            image: Input image (BGR format)
            detector: Detector instance to use
            confidence_threshold: Minimum confidence
            draw_boxes: Whether to draw bounding boxes
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        # Run detection
        detections = detector.detect(image, confidence_threshold)
        
        # Draw boxes if requested
        if draw_boxes:
            annotated = ImageProcessor.draw_detections(image, detections)
        else:
            annotated = image.copy()
        
        # Convert detections to dicts
        detections_dict = [d.to_dict() for d in detections]
        
        return annotated, detections_dict
    
    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format: str = "jpeg") -> str:
        """Encode image to base64 string."""
        import base64
        
        if format.lower() == "jpeg":
            _, buffer = cv2.imencode(".jpg", image)
        else:
            _, buffer = cv2.imencode(".png", image)
        
        return base64.b64encode(buffer).decode("utf-8")
    
    @staticmethod
    def decode_base64_to_image(base64_str: str) -> np.ndarray:
        """Decode base64 string to image."""
        import base64
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return image
    
    @staticmethod
    def calculate_statistics(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detection statistics."""
        if not detections:
            return {
                "total_objects": 0,
                "unique_classes": 0,
                "avg_confidence": 0,
                "class_counts": {},
                "has_pedestrians": False,
                "has_vehicles": False
            }
        
        # Count by class
        class_counts = {}
        total_confidence = 0
        
        for det in detections:
            class_name = det["class"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total_confidence += det["confidence"]
        
        # Check for pedestrians and vehicles
        pedestrian_classes = {"Pedestrian", "Person_sitting", "Cyclist"}
        vehicle_classes = {"Car", "Van", "Truck", "Tram"}
        
        has_pedestrians = any(c in pedestrian_classes for c in class_counts.keys())
        has_vehicles = any(c in vehicle_classes for c in class_counts.keys())
        
        return {
            "total_objects": len(detections),
            "unique_classes": len(class_counts),
            "avg_confidence": total_confidence / len(detections),
            "class_counts": class_counts,
            "has_pedestrians": has_pedestrians,
            "has_vehicles": has_vehicles
        }
