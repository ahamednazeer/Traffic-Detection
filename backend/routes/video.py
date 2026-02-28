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
import time
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from typing import Optional
from starlette.websockets import WebSocketState

from detectors import YOLODetector, YOLOCocoDetector, AccidentYOLODetector
from detectors.base_detector import Detection
from processors.image_processor import ImageProcessor
from utils.accident import summarize_accident_from_detections, select_top_accident_peaks

router = APIRouter(prefix="/api", tags=["video"])
VIDEO_JOBS = {}


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
        skip_frames = metadata.get("skip_frames", 0)
        preview_enabled = metadata.get("preview", True)
        preview_fps = metadata.get("preview_fps", 10)
        try:
            preview_fps = float(preview_fps)
        except (TypeError, ValueError):
            preview_fps = 10.0
        if preview_fps <= 0:
            preview_fps = 10.0
        preview_width = metadata.get("preview_width", 640)
        clip_seconds = metadata.get("clip_seconds", 8)
        try:
            clip_seconds = float(clip_seconds)
        except (TypeError, ValueError):
            clip_seconds = 8.0
        clip_seconds = min(max(clip_seconds, 4.0), 20.0)
        
        model_name = metadata.get("model", "accident_yolo11x") # Default to accident model
        job_id = metadata.get("job_id") or str(uuid.uuid4())
        effective_confidence = confidence

        VIDEO_JOBS[job_id] = {
            "job_id": job_id,
            "status": "processing",
            "progress": 0,
            "frame": 0,
            "total": 0,
            "preview": None,
            "result": None,
            "error": None
        }
        
        print(f"Video WebSocket: Receiving video ({total_size} bytes), Model: {model_name}")
        
        # Send acknowledgment
        await websocket.send_json({"type": "ready", "job_id": job_id})
        
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
        await websocket.send_json({"type": "status", "message": f"Loading model ({model_name})..."})
        
        # Select detector instance
        if model_name == "yolo":
            detector = YOLODetector()
        elif model_name == "accident_yolo11x":
            detector = AccidentYOLODetector()
        elif model_name.startswith("yolo11"):
            # Extract size (n, s, m, l, x)
            size = model_name[-1] if model_name[-1] in "nsmlx" else "m"
            detector = YOLOCocoDetector(model_size=size)
        elif model_name == "ensemble":
            # For video, ensemble is too slow, fallback to a good COCO model
            detector = YOLOCocoDetector(model_size="m")
        else:
            # Fallback
            detector = YOLOCocoDetector(model_size="m")
            
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
        VIDEO_JOBS[job_id]["total"] = total_frames
        effective_fps = fps / (skip_frames + 1) if fps > 0 else 0
        preview_stride = max(1, int(round(effective_fps / preview_fps))) if preview_enabled and effective_fps > 0 else 1
        min_preview_interval = 1.0 / preview_fps if preview_enabled and preview_fps > 0 else 0
        last_preview_time = 0.0

        async def send_preview(frame: np.ndarray, frame_index: int, processed_index: int):
            nonlocal last_preview_time
            if not preview_enabled:
                return
            if websocket.client_state != WebSocketState.CONNECTED:
                return
            if processed_index % preview_stride != 0 and processed_index != 1:
                return
            if min_preview_interval > 0:
                now = time.monotonic()
                if (now - last_preview_time) < min_preview_interval:
                    return
                last_preview_time = now
            preview_frame = frame
            if preview_width and preview_frame.shape[1] > preview_width:
                scale = preview_width / preview_frame.shape[1]
                new_h = max(1, int(preview_frame.shape[0] * scale))
                preview_frame = cv2.resize(preview_frame, (preview_width, new_h))
            ok, buffer = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            if not ok:
                return
            preview_base64 = base64.b64encode(buffer).decode('utf-8')
            VIDEO_JOBS[job_id]["preview"] = preview_base64
            try:
                await websocket.send_json({
                    "type": "preview",
                    "frame": preview_base64,
                    "frame_index": frame_index,
                    "processed_index": processed_index,
                    "total": total_frames
                })
            except Exception:
                return
        
        # Temp output
        temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        temp_raw_path = temp_raw.name
        temp_raw.close()
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_raw_path, fourcc, fps, (width, height))
        
        all_detections = []
        last_detections = []
        accident_timeline = []
        frame_count = 0
        processed_count = 0
        last_progress = 0
        best_accident_score = 0.0
        best_accident_frame = None
        best_accident_bbox = None
        best_accident_class = None
        client_disconnected = False
        
        await websocket.send_json({"type": "status", "message": "Processing frames..."})
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if websocket.client_state != WebSocketState.CONNECTED:
                client_disconnected = True
            progress = int((frame_count / total_frames) * 100)
            VIDEO_JOBS[job_id]["frame"] = frame_count
            VIDEO_JOBS[job_id]["progress"] = progress
            
            # Send progress every 5%
            if progress >= last_progress + 5:
                last_progress = progress
                if not client_disconnected:
                    try:
                        await websocket.send_json({
                            "type": "progress",
                            "progress": progress,
                            "frame": frame_count,
                            "total": total_frames
                        })
                    except Exception:
                        client_disconnected = True
                await asyncio.sleep(0)
            
            # Skip frames
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                if last_detections:
                    det_objects = [
                        Detection(
                            class_name=d.get("class") or d.get("class_name"),
                            confidence=d["confidence"],
                            bbox=(d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]),
                            class_id=d.get("class_id", -1)
                        )
                        for d in last_detections
                    ]
                    annotated_skip = ImageProcessor.draw_detections(frame, det_objects)
                    out.write(annotated_skip)
                else:
                    out.write(frame)
                continue
            
            # Process frame
            annotated, detections = await asyncio.to_thread(
                ImageProcessor.process_image,
                frame,
                detector,
                effective_confidence
            )
            
            all_detections.extend(detections)
            last_detections = detections
            processed_count += 1
            out.write(annotated)
            if not client_disconnected:
                await send_preview(annotated, frame_count, processed_count)

            accident = summarize_accident_from_detections(
                detections,
                confidence_threshold=effective_confidence
            )
            accident_timeline.append({
                "frame": frame_count,
                "timestamp": (frame_count / fps) if fps > 0 else 0,
                "score": accident["score"]
            })
            if accident["score"] > best_accident_score:
                best_accident_score = accident["score"]
                best_accident_frame = frame_count
                best_accident_bbox = accident.get("bbox")
                best_accident_class = accident.get("class_name")
        
        cap.release()
        out.release()

        if client_disconnected or websocket.client_state != WebSocketState.CONNECTED:
            print("Video WebSocket: Client disconnected, continuing processing for job polling.")
        
        print(f"Video WebSocket: Processed {processed_count} frames")
        
        # Encode
        if not client_disconnected:
            try:
                await websocket.send_json({
                    "type": "status",
                    "message": "Encoding video...",
                    "progress": 100
                })
            except Exception:
                client_disconnected = True
        
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
        
        stats = ImageProcessor.calculate_statistics(all_detections)
        
        print("Video WebSocket: Sending result")

        accident_peaks = select_top_accident_peaks(
            accident_timeline,
            max_peaks=3,
            min_separation_seconds=2.0
        )

        duration_seconds = total_frames / fps if fps > 0 else 0
        accident_clip = None
        if best_accident_score >= effective_confidence and best_accident_frame and fps > 0:
            best_timestamp = best_accident_frame / fps
            half = clip_seconds / 2
            start = max(0.0, best_timestamp - half)
            end = min(duration_seconds, best_timestamp + half)
            if end - start >= 1.0:
                temp_clip = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_clip_path = temp_clip.name
                temp_clip.close()
                try:
                    subprocess.run([
                        'ffmpeg', '-y', '-ss', f'{start:.3f}', '-to', f'{end:.3f}',
                        '-i', temp_output_path,
                        '-c:v', 'libx264', '-preset', 'fast',
                        '-crf', '23', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', '-an',
                        temp_clip_path
                    ], capture_output=True, check=True)
                    with open(temp_clip_path, 'rb') as f:
                        clip_base64 = base64.b64encode(f.read()).decode('utf-8')
                    accident_clip = {
                        "clip_base64": clip_base64,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "duration": round(end - start, 3),
                        "best_timestamp": round(best_timestamp, 3)
                    }
                except Exception as e:
                    print(f"Accident clip error: {e}")
                finally:
                    try:
                        os.unlink(temp_clip_path)
                    except Exception:
                        pass

        # Cleanup
        try:
            os.unlink(temp_input.name)
            os.unlink(temp_raw_path)
            os.unlink(temp_output_path)
        except Exception:
            pass

        accident_summary = {
            "detected": best_accident_score >= effective_confidence,
            "score": round(best_accident_score, 4),
            "threshold": effective_confidence,
            "class_name": best_accident_class,
            "bbox": best_accident_bbox,
            "best_frame": best_accident_frame,
            "best_timestamp": (best_accident_frame / fps) if best_accident_frame and fps > 0 else None
        }

        VIDEO_JOBS[job_id]["status"] = "complete"
        VIDEO_JOBS[job_id]["progress"] = 100
        VIDEO_JOBS[job_id]["result"] = {
            "video_base64": output_video,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "processed_frames": processed_count,
                "width": width,
                "height": height,
                "duration_seconds": duration_seconds
            },
            "statistics": stats,
            "accident": accident_summary,
            "accident_timeline": accident_timeline,
            "accident_peaks": accident_peaks,
            "accident_clip": accident_clip
        }

        if not client_disconnected:
            try:
                await websocket.send_json({
                    "type": "complete",
                    "video_base64": output_video,
                    "video_info": {
                        "fps": fps,
                        "total_frames": total_frames,
                        "processed_frames": processed_count,
                        "width": width,
                        "height": height,
                        "duration_seconds": duration_seconds
                    },
                    "statistics": stats,
                    "accident": accident_summary,
                    "accident_timeline": accident_timeline,
                    "accident_peaks": accident_peaks,
                    "accident_clip": accident_clip
                })
            except Exception:
                return
        
    except asyncio.TimeoutError:
        print("Video WebSocket: Timeout")
        await websocket.send_json({"error": "Timeout waiting for data"})
    except WebSocketDisconnect:
        print("Video WebSocket: Client disconnected")
    except Exception as e:
        print(f"Video WebSocket error: {e}")
        if "job_id" in locals():
            VIDEO_JOBS[job_id]["status"] = "error"
            VIDEO_JOBS[job_id]["error"] = str(e)
        import traceback
        traceback.print_exc()
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@router.get("/video/jobs/{job_id}")
async def get_video_job(job_id: str):
    job = VIDEO_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
