# Traffic Detection System

Real-time traffic sign and pedestrian detection using YOLO v11 and SSD models.

## Architecture

```
Traffic Detection/
├── backend/          # FastAPI Python backend
│   ├── main.py       # API entry point
│   ├── detectors/    # YOLO & SSD implementations
│   └── routes/       # API endpoints
│
├── frontend/         # Next.js 16 frontend
│   └── src/
│       ├── app/      # Pages
│       └── components/
│
└── best (8).pt       # YOLO model weights
```

## Quick Start

### 1. Start Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at: http://localhost:8000

### 2. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: http://localhost:3000

## Features

- **YOLO v11**: Fast, accurate detection
- **SSD300**: COCO pre-trained alternative
- **Ensemble Mode**: Combined detection with NMS
- **Image Detection**: Upload and analyze images
- **Video Processing**: Frame-by-frame analysis
- **Live Camera**: Real-time webcam detection (coming soon)

## Detection Classes

Car, Pedestrian, Van, Cyclist, Truck, Misc, Tram, Person_sitting

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/models` | List models |
| POST | `/api/models/select` | Select model |
| POST | `/api/detect/image` | Detect in image |
| POST | `/api/detect/video` | Process video |
