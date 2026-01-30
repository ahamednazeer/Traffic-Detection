"""
Video Processing WebSocket Routes
"""
import asyncio
import base64
import cv2
import numpy as np
import tempfile
import subprocess
import os
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Optional

from detectors import YOLODetector
from processors.image_processor import ImageProcessor

router = APIRouter(prefix="/api", tags=["video"])


@router.websocket("/video/process")
async def video_process_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for video processing with progress updates.
    """
    await websocket.accept()
    print("Video WebSocket: Connection accepted")
    
    try:
        # First message: metadata
        print("Video WebSocket: Waiting for metadata...")
        metadata_msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
        metadata = json.loads(metadata_msg)
        
        total_size = metadata.get("size", 0)
        confidence = metadata.get("confidence", 0.5)
        skip_frames = metadata.get("skip_frames", 2)
        
        print(f"Video WebSocket: Receiving video ({total_size} bytes)")
        
        # Send acknowledgment
        await websocket.send_json({"type": "ready"})
        
        # Receive video chunks
        chunks = []
        received = 0
        
        while received < total_size:
            chunk = await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
            chunks.append(chunk)
            received += len(base64.b64decode(chunk))
            
            # Send receive progress
            progress = int((received / total_size) * 100)
            await websocket.send_json({
                "type": "upload",
                "progress": progress
            })
        
        print(f"Video WebSocket: Received {received} bytes")
        
        # Decode and save video
        video_data = base64.b64decode(''.join(chunks))
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(video_data)
        temp_input.close()
        
        print(f"Video WebSocket: Saved to {temp_input.name}")
        
        # Load detector
        await websocket.send_json({"type": "status", "message": "Loading model..."})
        detector = YOLODetector()
        detector.load_model()
        
        # Open video
        cap = cv2.VideoCapture(temp_input.name)
        
        if not cap.isOpened():
            await websocket.send_json({"error": "Could not open video"})
            os.unlink(temp_input.name)
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video WebSocket: {total_frames} frames, {fps} fps, {width}x{height}")
        
        # Temp output
        temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_raw_path = temp_raw.name
        temp_raw.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        
        all_detections = []
        frame_count = 0
        processed_count = 0
        last_progress = 0
        
        await websocket.send_json({"type": "status", "message": "Processing frames..."})
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            progress = int((frame_count / total_frames) * 100)
            
            # Send progress every 5%
            if progress >= last_progress + 5:
                last_progress = progress
                await websocket.send_json({
                    "type": "progress",
                    "progress": progress,
                    "frame": frame_count,
                    "total": total_frames
                })
                await asyncio.sleep(0)
            
            # Skip frames
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                out.write(frame)
                continue
            
            # Process frame
            annotated, detections = ImageProcessor.process_image(
                frame, detector, confidence
            )
            
            all_detections.extend(detections)
            processed_count += 1
            out.write(annotated)
        
        cap.release()
        out.release()
        
        print(f"Video WebSocket: Processed {processed_count} frames")
        
        # Encode
        await websocket.send_json({
            "type": "status",
            "message": "Encoding video...",
            "progress": 100
        })
        
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_path = temp_output.name
        temp_output.close()
        
        try:
            subprocess.run([
                'ffmpeg', '-y', '-i', temp_raw_path,
                '-c:v', 'libx264', '-preset', 'fast',
                '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                temp_output_path
            ], capture_output=True, check=True)
        except Exception as e:
            print(f"FFmpeg error: {e}")
            import shutil
            shutil.copy(temp_raw_path, temp_output_path)
        
        # Read and encode output
        with open(temp_output_path, 'rb') as f:
            output_video = base64.b64encode(f.read()).decode('utf-8')
        
        # Cleanup
        os.unlink(temp_input.name)
        os.unlink(temp_raw_path)
        os.unlink(temp_output_path)
        
        stats = ImageProcessor.calculate_statistics(all_detections)
        
        print("Video WebSocket: Sending result")
        
        await websocket.send_json({
            "type": "complete",
            "video_base64": output_video,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "width": width,
                "height": height,
                "duration_seconds": total_frames / fps if fps > 0 else 0
            },
            "statistics": stats
        })
        
    except asyncio.TimeoutError:
        print("Video WebSocket: Timeout")
        await websocket.send_json({"error": "Timeout waiting for data"})
    except WebSocketDisconnect:
        print("Video WebSocket: Client disconnected")
    except Exception as e:
        print(f"Video WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass
