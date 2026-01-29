"""
Image Processing Utilities
"""
import cv2
import numpy as np
from typing import List, Tuple, Dict
import math
from collections import deque

from detectors.base_detector import BaseDetector, Detection
from utils.tracker import Tracker


class ImageProcessor:
    """Handles image processing and annotation."""
    
    _tracker = Tracker(max_disappeared=20, max_distance=100)
    # Store smoothed speed history: {id: deque(maxlen=5)}
    _speed_history: Dict[int, deque] = {} 
    
    # Class-based max speed limits (km/h) to filter noise
    MAX_SPEEDS = {
        "Person": 30,
        "Pedestrian": 20,
        "Cyclist": 45,
        "Person_sitting": 5,
        "Car": 200,
        "Truck": 160,
        "Bus": 140,
        "Motorcycle": 220
    }
    
    @staticmethod
    def get_ppm(y_coord: int, height: int) -> float:
        """
        Calculate dynamic Pixels Per Meter based on Y-coordinate (Depth).
        Assuming camera is looking down a road:
        - Bottom of frame (close): More pixels per meter
        - Top of frame (far): Fewer pixels per meter
        """
        # Simple linear interpolation calibration
        # Bottom of screen (y=height): ~100 PPM (close)
        # Top of road (y=height/2): ~10 PPM (far)
        
        # Normalize Y [0.0 to 1.0] from horizon line
        horizon = height * 0.3
        if y_coord < horizon:
            return 10.0 # Too far/horizon
            
        ratio = (y_coord - horizon) / (height - horizon)
        
        # PPM ranges from 15 (midscreen) to 140 (bottom) - Increased base values to lower speed est
        ppm = 15.0 + (ratio * 125.0)
        return max(10.0, ppm)

    @staticmethod
    def process_image(
        image: np.ndarray,
        detector: BaseDetector,
        confidence_threshold: float = 0.5,
        fps: float = 30.0
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process an image: detect objects, track them, estimate speed, and annotate.
        """
        height, width = image.shape[:2]
        
        # Run detection
        detections = detector.detect(image, confidence_threshold)
        
        # Prepare rectangles for tracker
        rects = []
        for det in detections:
            rects.append(det.bbox)
            
        # Update tracker
        objects = ImageProcessor._tracker.update(rects)
        
        # Convert detections to dictionary format
        detection_results = []
        annotated_image = image.copy()
        
        # Map IDs to speeds
        object_speed_map = {}
        
        for (object_id, centroid) in objects.items():
            # Calculate speed
            speed = ImageProcessor._calculate_speed(object_id, fps, height)
            object_speed_map[object_id] = speed
            
        # Draw annotations
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Find object ID for this detection (closest centroid)
            c_x = int((x1 + x2) / 2.0)
            c_y = int((y1 + y2) / 2.0)
            
            # Match detection to tracker ID
            matched_id = None
            min_dist = float('inf')
            
            for obj_id, centroid in objects.items():
                d = math.hypot(c_x - centroid[0], c_y - centroid[1])
                if d < 100:  # Search radius
                    if d < min_dist:
                        min_dist = d
                        matched_id = obj_id
            
            speed = 0.0
            if matched_id is not None:
                raw_speed = object_speed_map.get(matched_id, 0.0)
                
                # Apply class-specific limits
                max_allowed = ImageProcessor.MAX_SPEEDS.get(det.class_name, 250)
                speed = min(raw_speed, max_allowed)

            # Draw bounding box
            color = (0, 255, 0) # Green default
            if speed > 80: color = (0, 0, 255) # Red for fast
            elif speed > 50: color = (0, 165, 255) # Orange for medium
            
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Label
            label = f"{det.class_name} {det.confidence:.2f}"
            if speed > 3:
                label += f" | {int(speed)} km/h"
                
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_image, (x1, y1 - 25), (x1 + w, y1), color, -1)
            
            cv2.putText(
                annotated_image,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1
            )
            
            detection_results.append({
                "class_name": det.class_name,
                "confidence": det.confidence,
                "bbox": det.bbox,
                "class_id": det.class_id,
                "track_id": matched_id,
                "speed": speed
            })
            
        return annotated_image, detection_results

    @staticmethod
    def _calculate_speed(object_id: int, fps: float, height: int) -> float:
        """
        Estimate speed in km/h using variable PPM over 5-frame average.
        """
        history = ImageProcessor._tracker.history.get(object_id, [])
        if len(history) < 3: # Need at least 3 points for smooth speed
            return 0.0
            
        # Initialize deque if needed
        if object_id not in ImageProcessor._speed_history:
            ImageProcessor._speed_history[object_id] = deque(maxlen=8)
            
        # Get movement between frames
        # Use average of last 3 movements to reduce jitter
        points = list(history)[-4:] # Get last 4 points
        
        speeds = []
        for i in range(1, len(points)):
            p1 = points[i-1]
            p2 = points[i]
            
            # Pixel distance
            d_pixels = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            
            # Dynamic PPM based on average Y position
            avg_y = (p1[1] + p2[1]) / 2.0
            ppm = ImageProcessor.get_ppm(avg_y, height)
            
            # Convert to meters
            d_meters = d_pixels / ppm
            
            # Speed in km/h
            frame_speed = (d_meters * fps) * 3.6
            speeds.append(frame_speed)
            
        if not speeds:
            return 0.0
            
        # Current instant speed (avg of last few frames)
        current_speed = sum(speeds) / len(speeds)
        
        # Add to long-term history
        ImageProcessor._speed_history[object_id].append(current_speed)
        
        # Return rolling average speed
        avg_speed = sum(ImageProcessor._speed_history[object_id]) / len(ImageProcessor._speed_history[object_id])
        return avg_speed

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
            name = det["class_name"]
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
