"""
Image Processing Utilities
"""
import cv2
import numpy as np
import base64
from typing import List, Tuple, Dict

from detectors.base_detector import BaseDetector, Detection


class ImageProcessor:
    """Handles image processing and annotation."""

    @staticmethod
    def process_image(
        image: np.ndarray,
        detector: BaseDetector,
        confidence_threshold: float = 0.5,
        draw_boxes: bool = True
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process an image: detect objects and annotate.
        """
        # Run detection
        detections = detector.detect(image, confidence_threshold)
        
        # Convert detections to dictionary format using the built-in to_dict
        detection_results = [det.to_dict() for det in detections]
        
        # Draw annotations if requested
        annotated_image = image.copy()
        if draw_boxes:
            annotated_image = ImageProcessor.draw_detections(annotated_image, detections)
            
        return annotated_image, detection_results

    @staticmethod
    def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw detections on the image."""
        annotated_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Use class colors from settings
            from config.settings import CLASS_COLORS
            color = CLASS_COLORS.get(det.class_name, (0, 255, 0))
            
            # Draw thicker bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 3)
            
            # Prepare label
            label = f"{det.class_name} {det.confidence:.0%}"
            
            # Use larger font for better visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            
            # Get text size
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
            
            # Position label above box (or inside if at top edge)
            label_y = y1 - 10 if y1 > 35 else y1 + text_h + 10
            label_x = x1
            
            # Draw background rectangle with padding
            padding = 5
            bg_y1 = label_y - text_h - padding
            bg_y2 = label_y + padding
            bg_x1 = label_x - padding
            bg_x2 = label_x + text_w + padding
            
            # Semi-transparent dark background
            overlay = annotated_image.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
            
            # Draw colored border around label
            cv2.rectangle(annotated_image, (bg_x1, bg_y1), (bg_x2, bg_y2), color, 2)
            
            # Draw text with outline for better visibility
            cv2.putText(annotated_image, label, (label_x, label_y), font, font_scale, (0, 0, 0), font_thickness + 2)
            cv2.putText(annotated_image, label, (label_x, label_y), font, font_scale, (255, 255, 255), font_thickness)
            
        return annotated_image

    @staticmethod
    def calculate_statistics(detections: List[Dict]) -> Dict:
        """Calculate complete statistics for frontend."""
        total_objects = len(detections)
        class_counts = {}
        unique_classes = set()
        confidence_sum = 0.0
        has_pedestrians = False
        has_vehicles = False
        
        vehicle_classes = ["Car", "Truck", "Van", "Cyclist", "Tram", "Motorcycle", "Bus"]
        pedestrian_classes = ["Pedestrian", "Person", "Person_sitting"]
        
        for det in detections:
            # Note: detection dict from det.to_dict() uses key "class"
            name = det.get("class") or det.get("class_name")
            class_counts[name] = class_counts.get(name, 0) + 1
            unique_classes.add(name)
            confidence_sum += det["confidence"]
            
            if name in pedestrian_classes:
                has_pedestrians = True
            if name in vehicle_classes:
                has_vehicles = True
        
        avg_confidence = confidence_sum / total_objects if total_objects > 0 else 0
        
        return {
            "total_objects": total_objects,
            "unique_classes": len(unique_classes),
            "avg_confidence": avg_confidence,
            "class_counts": class_counts,
            "has_pedestrians": has_pedestrians,
            "has_vehicles": has_vehicles
        }

    @staticmethod
    def encode_image_to_base64(image: np.ndarray, format: str = ".jpg") -> str:
        """
        Encode an image to a base64 data URL.
        
        Args:
            image: Input image as numpy array
            format: Image format extension (e.g., '.jpg', '.png')
            
        Returns:
            Base64 encoded data URL string
        """
        success, encoded = cv2.imencode(format, image)
        if not success:
            raise ValueError("Failed to encode image")
        
        base64_bytes = base64.b64encode(encoded)
        base64_string = base64_bytes.decode('utf-8')
        
        mime_type = "image/jpeg" if format == ".jpg" else "image/png"
        return f"data:{mime_type};base64,{base64_string}"
