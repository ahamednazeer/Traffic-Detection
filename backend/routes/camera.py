"""
WebSocket Camera Streaming Routes
"""
import asyncio
import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional

from detectors import YOLODetector, SSDDetector
from processors.image_processor import ImageProcessor

router = APIRouter(prefix="/api", tags=["camera"])

# Global detector instances
_camera_detector: Optional[YOLODetector] = None


def get_camera_detector():
    """Get or create detector for camera."""
    global _camera_detector
    if _camera_detector is None:
        _camera_detector = YOLODetector()
        _camera_detector.load_model()
    return _camera_detector


@router.websocket("/camera")
async def camera_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for real-time camera detection.
    
    Client sends: base64 encoded JPEG frames
    Server responds: JSON with annotated frame and detections
    """
    await websocket.accept()
    
    detector = get_camera_detector()
    
    try:
        while True:
            # Receive frame from client
            data = await websocket.receive_text()
            
            try:
                # Decode base64 image
                img_data = base64.b64decode(data)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_json({"error": "Invalid frame"})
                    continue
                
                # Run detection
                annotated, detections = ImageProcessor.process_image(
                    frame, detector, confidence_threshold=0.5
                )
                
                # Encode annotated frame
                _, buffer = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
                annotated_base64 = base64.b64encode(buffer).decode('utf-8')
                
                # Calculate stats
                stats = ImageProcessor.calculate_statistics(detections)
                
                # Send response
                await websocket.send_json({
                    "frame": annotated_base64,
                    "detections": detections,
                    "stats": stats
                })
                
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("Camera WebSocket disconnected")
    except Exception as e:
        print(f"Camera WebSocket error: {e}")
