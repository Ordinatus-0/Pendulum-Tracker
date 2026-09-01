"""YOLOv8 bob detector with a conservative contour fallback candidate selector."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    width: float
    height: float
    confidence: float


class YOLOBobDetector:
    """Wrap Ultralytics so a custom pendulum model can replace the default weights.

    COCO models do not contain a dedicated 'pendulum bob' class.  Therefore the
    default model is useful only where the bob resembles a known object; a custom
    model trained with a single ``bob`` class is the reliable option.
    """

    def __init__(self, weights: str = "yolov8n.pt", confidence: float = 0.25, bob_class: int | None = None):
        self.weights = weights
        self.confidence = confidence
        self.bob_class = bob_class
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Ultralytics is not installed. Install requirements.txt to run YOLO detection.") from exc
            self._model = YOLO(self.weights)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._load_model()
        classes = [self.bob_class] if self.bob_class is not None else None
        result = model.predict(frame, conf=self.confidence, classes=classes, verbose=False)[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xywh.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        return [Detection(*map(float, box), float(score)) for box, score in zip(boxes, scores)]

    @staticmethod
    def select_candidate(candidates: Iterable[Detection], previous: Detection | None) -> Detection | None:
        """Select nearest plausible candidate to preserve identity across frames."""
        candidates = list(candidates)
        if not candidates:
            return None
        if previous is None:
            return max(candidates, key=lambda item: item.confidence)
        return min(candidates, key=lambda item: (item.x - previous.x) ** 2 + (item.y - previous.y) ** 2 - 5000 * item.confidence)
