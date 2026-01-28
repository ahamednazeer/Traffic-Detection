"""
Download state tracking - simplified (no progress tracking)
"""

_download_state = {
    "is_downloading": False,
    "model_name": "",
    "progress": 0
}


def set_download_state(is_downloading: bool, model_name: str = "", progress: int = 0):
    """Set download state."""
    _download_state["is_downloading"] = is_downloading
    _download_state["model_name"] = model_name
    _download_state["progress"] = progress


def reset_download_state():
    """Reset download state."""
    _download_state["is_downloading"] = False
    _download_state["model_name"] = ""
    _download_state["progress"] = 0


def get_download_state():
    """Get current download state."""
    return _download_state.copy()
