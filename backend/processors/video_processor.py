"""
Video Processing Utilities
"""
import cv2
import numpy as np
import tempfile
import subprocess
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
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {"error": "Could not open video file"}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Create temp file for raw output
        temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_raw_path = temp_raw.name
        temp_raw.close()
        
        # Use MJPG codec for temp file (more compatible)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        
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
                out.write(frame)
                continue
            
            # Process frame
            annotated, detections = ImageProcessor.process_image(
                frame, self.detector, confidence_threshold
            )
            
            all_detections.extend(detections)
            processed_count += 1
            out.write(annotated)
        
        cap.release()
        out.release()
        
        # Convert to H.264 using ffmpeg for browser compatibility
        if output_path:
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_raw_path,
                    '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '23', '-pix_fmt', 'yuv420p',
                    '-movflags', '+faststart',
                    output_path
                ], capture_output=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"FFmpeg error: {e.stderr.decode()}")
                # Fallback: just copy the raw file
                import shutil
                shutil.copy(temp_raw_path, output_path)
            except FileNotFoundError:
                # ffmpeg not installed, use raw file
                import shutil
                shutil.copy(temp_raw_path, output_path)
            
            # Cleanup temp file
            try:
                os.unlink(temp_raw_path)
            except:
                pass
        
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
