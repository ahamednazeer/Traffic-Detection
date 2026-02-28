"""
Accident YOLO detector (pretrained weights).
"""
from pathlib import Path
from typing import List
import urllib.request

import numpy as np
from ultralytics import YOLO

from .base_detector import BaseDetector, Detection
from config.settings import ACCIDENT_MODEL_PATH, ACCIDENT_MODEL_URL
from utils.download import set_download_state, reset_download_state


class AccidentYOLODetector(BaseDetector):
    """YOLO detector for accident detection using pretrained weights."""

    def __init__(self, model_path: str = None):
        super().__init__(model_path or str(ACCIDENT_MODEL_PATH))
        self._class_names: List[str] = []

    def _ensure_model_file(self) -> None:
        model_path = Path(self.model_path)
        if model_path.exists():
            return

        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not ACCIDENT_MODEL_URL:
            raise RuntimeError("ACCIDENT_MODEL_URL is not set.")

        set_download_state(is_downloading=True, model_name="Accident (Custom)", progress=0)
        try:
            urllib.request.urlretrieve(ACCIDENT_MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download accident model weights. "
                f"Download manually to {model_path}."
            ) from exc
        finally:
            reset_download_state()

    def load_model(self) -> bool:
        try:
            self._ensure_model_file()
            self.model = YOLO(self.model_path)
            self.is_loaded = True

            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self._class_names = [names[k] for k in sorted(names.keys())]
            elif isinstance(names, list):
                self._class_names = names
            else:
                self._class_names = ["accident", "vehicle"]

            print(f"Accident YOLO model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"Failed to load accident model: {e}")
            self.is_loaded = False
            return False

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: float = 0.5
    ) -> List[Detection]:
        if not self.is_loaded:
            if not self.load_model():
                return []

        detections: List[Detection] = []
        try:
            results = self.model(
                image,
                conf=confidence_threshold,
                imgsz=960,
                verbose=False,
                save=False,
                save_txt=False,
                save_conf=False
            )
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    if class_id < len(self._class_names):
                        class_name = self._class_names[class_id]
                    else:
                        class_name = f"Class_{class_id}"

                    detections.append(Detection(
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id
                    ))
        except Exception as e:
            print(f"Accident YOLO detection error: {e}")

        return detections

    def get_model_name(self) -> str:
        return "Accident (Custom)"

    def get_class_names(self) -> List[str]:
        return self._class_names or ["accident", "vehicle"]
