"""
Video Processing Utilities
"""
import cv2
import numpy as np
import tempfile
import os
from typing import List, Dict, Any, Generator, Tuple
from detectors.base_detector import BaseDetector
from .image_processor import ImageProcessor


class VideoProcessor:
    """Handles video processing for detection."""
    
    def __init__(self, detector: BaseDetector):
        self.detector = detector
    
    def process_video_file(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        output_path: str = None,
        skip_frames: int = 0
    ) -> Dict[str, Any]:
        """
        Process a video file and return detection results.
        
        Args:
            video_path: Path to input video
            confidence_threshold: Minimum detection confidence
            output_path: Optional path for annotated output video
            skip_frames: Number of frames to skip between detections
            
        Returns:
            Dictionary with processing results and statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Setup output video if requested
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        processed_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                if out:
                    out.write(frame)
                continue
            
            # Process frame
            annotated, detections = ImageProcessor.process_image(
                frame, self.detector, confidence_threshold
            )
            
            all_detections.extend(detections)
            processed_count += 1
            
            if out:
                out.write(annotated)
        
        cap.release()
        if out:
            out.release()
        
        # Calculate statistics
        stats = ImageProcessor.calculate_statistics(all_detections)
        
        return {
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "width": width,
                "height": height,
                "duration_seconds": duration
            },
            "statistics": stats,
            "detections": all_detections,
            "output_path": output_path
        }
    
    def process_video_stream(
        self,
        video_path: str,
        confidence_threshold: float = 0.5,
        skip_frames: int = 0
    ) -> Generator[Tuple[np.ndarray, List[Dict[str, Any]], float], None, None]:
        """
        Process video as a stream, yielding frames with detections.
        
        Yields:
            Tuple of (annotated_frame, detections, progress)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = frame_count / total_frames if total_frames > 0 else 0
            
            # Skip frames if requested
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                continue
            
            # Process frame
            annotated, detections = ImageProcessor.process_image(
                frame, self.detector, confidence_threshold
            )
            
            yield annotated, detections, progress
        
        cap.release()
