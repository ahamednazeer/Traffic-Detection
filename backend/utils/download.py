"""
Model Download Utilities with Progress Tracking
"""
import os
import urllib.request
from pathlib import Path
from typing import Callable, Optional
import threading

# Global download state
_download_state = {
    "is_downloading": False,
    "model_name": "",
    "progress": 0,
    "total_size": 0,
    "downloaded": 0,
    "error": None
}
_state_lock = threading.Lock()


def get_download_state() -> dict:
    """Get current download state."""
    with _state_lock:
        return _download_state.copy()


def set_download_state(
    is_downloading: bool = None,
    model_name: str = None,
    progress: float = None,
    total_size: int = None,
    downloaded: int = None,
    error: str = None
):
    """Update download state."""
    with _state_lock:
        if is_downloading is not None:
            _download_state["is_downloading"] = is_downloading
        if model_name is not None:
            _download_state["model_name"] = model_name
        if progress is not None:
            _download_state["progress"] = progress
        if total_size is not None:
            _download_state["total_size"] = total_size
        if downloaded is not None:
            _download_state["downloaded"] = downloaded
        if error is not None:
            _download_state["error"] = error


def download_with_progress(
    url: str,
    destination: Path,
    model_name: str,
    progress_callback: Optional[Callable[[float], None]] = None
) -> bool:
    """
    Download a file with progress tracking.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        model_name: Name of the model (for display)
        progress_callback: Optional callback for progress updates
        
    Returns:
        True if successful
    """
    try:
        set_download_state(
            is_downloading=True,
            model_name=model_name,
            progress=0,
            downloaded=0,
            error=None
        )
        
        # Create directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Get file info
        response = urllib.request.urlopen(url)
        total_size = int(response.headers.get('content-length', 0))
        set_download_state(total_size=total_size)
        
        # Download with progress
        block_size = 8192
        downloaded = 0
        
        with open(destination, 'wb') as f:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                    
                downloaded += len(buffer)
                f.write(buffer)
                
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    set_download_state(progress=progress, downloaded=downloaded)
                    
                    if progress_callback:
                        progress_callback(progress)
        
        set_download_state(is_downloading=False, progress=100)
        return True
        
    except Exception as e:
        set_download_state(is_downloading=False, error=str(e))
        return False


def reset_download_state():
    """Reset download state to initial values."""
    with _state_lock:
        _download_state["is_downloading"] = False
        _download_state["model_name"] = ""
        _download_state["progress"] = 0
        _download_state["total_size"] = 0
        _download_state["downloaded"] = 0
        _download_state["error"] = None
