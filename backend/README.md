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
- `WS /api/video/process` - Video processing with live preview and timeline
- `GET /api/video/jobs/{job_id}` - Poll video job status/results

## Accident Model Weights

The accident model auto-downloads on first load. If you want to download manually, place the file at:

```
backend/models/accident_train/weights/best.pt
```

## Email Alert Configuration

Set these environment variables on backend to enable industry-style email alerts when a video detects an accident:

```bash
ALERT_EMAIL_ENABLED=true
SMTP_HOST=smtp.your-provider.com
SMTP_PORT=587
SMTP_USERNAME=alerts@company.com
SMTP_PASSWORD=your_app_password
SMTP_FROM_EMAIL=alerts@company.com
SMTP_FROM_NAME="Traffic Detection System"
SMTP_USE_TLS=true
```
