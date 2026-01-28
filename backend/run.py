#!/usr/bin/env python3
"""
Traffic Detection Backend - Run Script
"""
import sys
import os

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn


def main():
    """Run the FastAPI server."""
    print("Starting Traffic Detection Backend...")
    print("API docs: http://localhost:8000/docs")
    print("Health: http://localhost:8000/api/health")
    print("-" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
