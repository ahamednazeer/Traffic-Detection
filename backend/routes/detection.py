"""
Detection API Routes
"""
import cv2
import numpy as np
import tempfile
import os
import base64
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from detectors import YOLODetector, YOLOCocoDetector, SSDDetector, BaseDetector
from processors.image_processor import ImageProcessor
from processors.video_processor import VideoProcessor
from config.settings import CLASS_COLORS, CLASS_NAMES


router = APIRouter(prefix="/api", tags=["detection"])

# Global detector instances
_detectors = {
    "yolo": None,
    "yolo11n": None,  # nano
    "yolo11s": None,  # small
    "yolo11m": None,  # medium
    "yolo11l": None,  # large
    "yolo11x": None,  # xlarge
    "ssd": None
}
_active_model = "yolo11x"  # Default to xlarge for best accuracy


class DetectionRequest(BaseModel):
    confidence: float = 0.5
    model: str = "yolo11x"


class ModelSelectRequest(BaseModel):
    model: str


def get_detector(model_name: str) -> BaseDetector:
    """Get or create a detector instance."""
    global _detectors
    
    if model_name == "yolo":
        if _detectors["yolo"] is None:
            _detectors["yolo"] = YOLODetector()
            _detectors["yolo"].load_model()
        return _detectors["yolo"]
    
    elif model_name == "yolo11n":
        if _detectors["yolo11n"] is None:
            _detectors["yolo11n"] = YOLOCocoDetector(model_size="n")
            _detectors["yolo11n"].load_model()
        return _detectors["yolo11n"]
    
    elif model_name == "yolo11s":
        if _detectors["yolo11s"] is None:
            _detectors["yolo11s"] = YOLOCocoDetector(model_size="s")
            _detectors["yolo11s"].load_model()
        return _detectors["yolo11s"]
    
    elif model_name == "yolo11m":
        if _detectors["yolo11m"] is None:
            _detectors["yolo11m"] = YOLOCocoDetector(model_size="m")
            _detectors["yolo11m"].load_model()
        return _detectors["yolo11m"]
    
    elif model_name == "yolo11l":
        if _detectors["yolo11l"] is None:
            _detectors["yolo11l"] = YOLOCocoDetector(model_size="l")
            _detectors["yolo11l"].load_model()
        return _detectors["yolo11l"]
    
    elif model_name == "yolo11x":
        if _detectors["yolo11x"] is None:
            _detectors["yolo11x"] = YOLOCocoDetector(model_size="x")
            _detectors["yolo11x"].load_model()
        return _detectors["yolo11x"]
    
    elif model_name == "ssd":
        if _detectors["ssd"] is None:
            _detectors["ssd"] = SSDDetector()
            _detectors["ssd"].load_model()
        return _detectors["ssd"]
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def merge_detections(det1: list, det2: list, iou_threshold: float = 0.5) -> list:
    """Merge detections from two models using NMS."""
    if not det1:
        return det2
    if not det2:
        return det1
    
    all_dets = det1 + det2
    
    # Simple NMS based on IoU
    keep = []
    used = set()
    
    # Sort by confidence
    sorted_dets = sorted(enumerate(all_dets), key=lambda x: x[1]["confidence"], reverse=True)
    
    for idx, det in sorted_dets:
        if idx in used:
            continue
        
        keep.append(det)
        used.add(idx)
        
        # Mark overlapping detections as used
        for idx2, det2 in sorted_dets:
            if idx2 in used:
                continue
            
            # Calculate IoU
            b1 = det["bbox"]
            b2 = det2["bbox"]
            
            x1 = max(b1["x1"], b2["x1"])
            y1 = max(b1["y1"], b2["y1"])
            x2 = min(b1["x2"], b2["x2"])
            y2 = min(b1["y2"], b2["y2"])
            
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
            area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
            union = area1 + area2 - inter
            
            iou = inter / union if union > 0 else 0
            
            if iou > iou_threshold:
                used.add(idx2)
    
    return keep


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "active_model": _active_model}


@router.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": [
            {
                "id": "yolo",
                "name": "YOLO (Custom)",
                "description": "Custom-trained for traffic (8 classes)",
                "size": "~25MB",
                "loaded": _detectors["yolo"] is not None and _detectors["yolo"].is_loaded
            },
            {
                "id": "yolo11n",
                "name": "YOLO11 Nano",
                "description": "Fastest, basic accuracy",
                "size": "~5MB",
                "loaded": _detectors["yolo11n"] is not None and _detectors["yolo11n"].is_loaded
            },
            {
                "id": "yolo11s",
                "name": "YOLO11 Small",
                "description": "Fast, good accuracy",
                "size": "~18MB",
                "loaded": _detectors["yolo11s"] is not None and _detectors["yolo11s"].is_loaded
            },
            {
                "id": "yolo11m",
                "name": "YOLO11 Medium",
                "description": "Balanced speed/accuracy",
                "size": "~40MB",
                "loaded": _detectors["yolo11m"] is not None and _detectors["yolo11m"].is_loaded
            },
            {
                "id": "yolo11l",
                "name": "YOLO11 Large",
                "description": "High accuracy",
                "size": "~75MB",
                "loaded": _detectors["yolo11l"] is not None and _detectors["yolo11l"].is_loaded
            },
            {
                "id": "yolo11x",
                "name": "YOLO11 XLarge",
                "description": "Best accuracy (default)",
                "size": "~140MB",
                "loaded": _detectors["yolo11x"] is not None and _detectors["yolo11x"].is_loaded
            },
            {
                "id": "ssd",
                "name": "SSD300",
                "description": "VGG16 backbone, COCO",
                "size": "~100MB",
                "loaded": _detectors["ssd"] is not None and _detectors["ssd"].is_loaded
            },
            {
                "id": "ensemble",
                "name": "Ensemble",
                "description": "YOLO + SSD combined",
                "size": "Multi",
                "loaded": False
            }
        ],
        "active": _active_model,
        "class_names": CLASS_NAMES,
        "class_colors": {k: f"rgb({v[2]},{v[1]},{v[0]})" for k, v in CLASS_COLORS.items()}
    }


@router.post("/models/select")
async def select_model(request: ModelSelectRequest):
    """Select the active model."""
    global _active_model
    
    valid_models = ["yolo", "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x", "ssd", "ensemble"]
    if request.model not in valid_models:
        raise HTTPException(status_code=400, detail=f"Invalid model. Valid options: {valid_models}")
    
    _active_model = request.model
    
    # Pre-load the selected model
    if request.model == "ensemble":
        get_detector("yolo")
        get_detector("ssd")
    elif request.model != "ensemble":
        get_detector(request.model)
    
    return {"status": "success", "active_model": _active_model}


@router.post("/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    model: str = Form(None)
):
    """Detect objects in an uploaded image."""
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Use specified model or active model
        model_to_use = model or _active_model
        
        if model_to_use == "ensemble":
            # Run both detectors
            yolo_detector = get_detector("yolo")
            ssd_detector = get_detector("ssd")
            
            _, yolo_dets = ImageProcessor.process_image(image, yolo_detector, confidence, draw_boxes=False)
            _, ssd_dets = ImageProcessor.process_image(image, ssd_detector, confidence, draw_boxes=False)
            
            # Merge detections
            detections = merge_detections(yolo_dets, ssd_dets)
            
            # Draw merged detections
            from detectors.base_detector import Detection
            det_objects = [
                Detection(
                    class_name=d["class"],
                    confidence=d["confidence"],
                    bbox=(d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]),
                    class_id=d["class_id"]
                )
                for d in detections
            ]
            annotated = ImageProcessor.draw_detections(image, det_objects)
        else:
            detector = get_detector(model_to_use)
            annotated, detections = ImageProcessor.process_image(image, detector, confidence)
        
        # Calculate statistics
        stats = ImageProcessor.calculate_statistics(detections)
        
        # Encode annotated image
        annotated_base64 = ImageProcessor.encode_image_to_base64(annotated)
        
        return {
            "success": True,
            "model_used": model_to_use,
            "annotated_image": annotated_base64,
            "detections": detections,
            "statistics": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    confidence: float = Form(0.5),
    model: str = Form(None),
    skip_frames: int = Form(0)
):
    """Process a video file for detection."""
    try:
        # Save uploaded video to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            contents = await file.read()
            tmp.write(contents)
            video_path = tmp.name
        
        # Create temp output path
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        
        # Use specified model or active model
        model_to_use = model or _active_model
        
        if model_to_use == "ensemble":
            # For video, just use YOLO for speed (ensemble is too slow for video)
            model_to_use = "yolo"
        
        detector = get_detector(model_to_use)
        processor = VideoProcessor(detector)
        
        # Process video
        result = processor.process_video_file(
            video_path,
            confidence_threshold=confidence,
            output_path=output_path,
            skip_frames=skip_frames
        )
        
        # Read output video and encode
        with open(output_path, "rb") as f:
            video_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup temp files
        os.unlink(video_path)
        os.unlink(output_path)
        
        return {
            "success": True,
            "model_used": model_to_use,
            "video_base64": video_data,
            "video_info": result["video_info"],
            "statistics": result["statistics"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
