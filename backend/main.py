"""
Traffic Detection Backend - FastAPI Application
"""
import sys
from pathlib import Path

# Add backend directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import CORS_ORIGINS, API_PREFIX
from routes.detection import router as detection_router
from routes.camera import router as camera_router
from routes.video import router as video_router


# Create FastAPI app
app = FastAPI(
    title="Traffic Detection API",
    description="Real-time traffic sign and pedestrian detection using YOLO v11 and SSD",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(detection_router)
app.include_router(camera_router)
app.include_router(video_router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Traffic Detection API",
        "version": "1.0.0",
        "description": "YOLO v11 + SSD Traffic Detection",
        "docs": "/docs",
        "health": "/api/health"
    }


@app.on_event("startup")
async def startup_event():
    """Pre-load YOLO model on startup."""
    print("🚀 Starting Traffic Detection API...")
    
    # Pre-load YOLO model
    try:
        from routes.detection import get_detector
        get_detector("yolo")
        print("✅ YOLO model pre-loaded")
    except Exception as e:
        print(f"⚠️ Could not pre-load YOLO model: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
