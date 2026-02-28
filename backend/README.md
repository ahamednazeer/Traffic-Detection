# Traffic Detection Backend

Python FastAPI backend for traffic detection with YOLO v11 and SSD models.
Now includes a custom accident detection model.

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
- `GET /api/datasets` - List available pretrained accident datasets
- `POST /api/datasets/select` - Select active pretrained dataset
- `WS /api/camera` - WebSocket for camera stream

## Accident Model Weights

The accident model auto-downloads on first load. If you want to download manually, place the file at:

```
backend/models/accident_train/weights/best.pt
```
