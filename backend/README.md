# Traffic Detection Backend

Python FastAPI backend for traffic detection with YOLO v11 and SSD models.

## Setup

```bash
cd backend
pip install -r requirements.txt
python3 run.py
```

## API Endpoints

- `POST /api/detect/image` - Detect objects in image
- `POST /api/detect/video` - Process video file
- `GET /api/models` - List available models
- `POST /api/models/select` - Select active model
- `WS /api/camera` - WebSocket for camera stream
