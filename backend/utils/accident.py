"""
Accident detection utilities.
"""
from typing import Dict, List

ACCIDENT_CLASS_ALIASES = {
    "accident",
    "collision",
    "crash",
    "car crash",
    "traffic accident",
}

DATASET_THRESHOLDS = {
    "cadp": 0.15,
    "tumtraf-accid3nd": 0.2,
}


def get_dataset_threshold(dataset_id: str, fallback: float = 0.2) -> float:
    return DATASET_THRESHOLDS.get(dataset_id, fallback)


def summarize_accident_from_detections(
    detections: List[Dict],
    confidence_threshold: float = 0.5
) -> Dict:
    """
    Summarize accident detections from model outputs.
    Uses class names like "accident", "collision", etc.
    """
    best_score = 0.0
    best_bbox = None
    best_class = None

    for det in detections:
        name = (det.get("class") or det.get("class_name") or "").strip().lower()
        if name in ACCIDENT_CLASS_ALIASES:
            score = float(det.get("confidence", 0.0))
            if score > best_score:
                best_score = score
                best_class = name
                best_bbox = det.get("bbox")

    detected = best_score >= confidence_threshold

    return {
        "detected": detected,
        "score": round(best_score, 4),
        "threshold": confidence_threshold,
        "class_name": best_class,
        "bbox": best_bbox
    }


def select_top_accident_peaks(
    timeline: List[Dict],
    max_peaks: int = 3,
    min_separation_seconds: float = 2.0
) -> List[Dict]:
    """
    Pick top accident score peaks with a minimum time separation.
    Expects timeline entries with "timestamp" and "score".
    """
    if not timeline:
        return []

    sorted_entries = sorted(timeline, key=lambda x: x.get("score", 0.0), reverse=True)
    peaks: List[Dict] = []

    for entry in sorted_entries:
        score = float(entry.get("score", 0.0))
        if score <= 0:
            break
        timestamp = float(entry.get("timestamp", 0.0))
        if any(abs(timestamp - float(p.get("timestamp", 0.0))) < min_separation_seconds for p in peaks):
            continue
        peaks.append({
            "frame": entry.get("frame"),
            "timestamp": round(timestamp, 3),
            "score": round(score, 4)
        })
        if len(peaks) >= max_peaks:
            break

    return peaks
