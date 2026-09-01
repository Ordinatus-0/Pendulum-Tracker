"""Video validation, frame processing, interpolation, and overlay rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

from app.detection.yolo_detector import Detection, YOLOBobDetector
from app.models import ProcessingResult, VideoMetadata

VALID_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


def video_metadata(path: str | Path) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video. Use a readable MP4, MOV, AVI, or MKV file.")
    metadata = VideoMetadata(
        fps=float(capture.get(cv2.CAP_PROP_FPS) or 0),
        frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    )
    capture.release()
    if metadata.fps <= 0 or metadata.frame_count <= 0:
        raise ValueError("The uploaded file has no readable video frames or FPS metadata.")
    return metadata


def validate_video(path: str | Path) -> VideoMetadata:
    path = Path(path)
    if path.suffix.lower() not in VALID_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{path.suffix}'. Supported formats: {', '.join(sorted(VALID_EXTENSIONS))}.")
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError("The uploaded video is empty or unavailable.")
    return video_metadata(path)


def interpolate_missing(data: pd.DataFrame, max_gap_frames: int = 10) -> pd.DataFrame:
    """Fill only short internal gaps and preserve unresolved missing measurements.

    A long occlusion, or a gap at either end, has no measured bounding points on
    both sides and must remain missing rather than being presented as data.
    """
    result = data.copy()
    coordinate_columns = ["x_px", "y_px", "bbox_width_px", "bbox_height_px"]
    for column in coordinate_columns:
        result[f"raw_{column}"] = result[column]
    result["raw_confidence"] = result["confidence"]
    missing = result["tracking_status"].eq("missing").to_numpy()
    starts = np.flatnonzero(missing & np.r_[True, ~missing[:-1]])
    ends = np.flatnonzero(missing & np.r_[~missing[1:], True])
    for start, end in zip(starts, ends):
        gap_size = end - start + 1
        has_bracketing_measurements = start > 0 and end < len(result) - 1
        if gap_size <= max_gap_frames and has_bracketing_measurements:
            for column in coordinate_columns:
                before, after = result.at[start - 1, column], result.at[end + 1, column]
                result.loc[start:end, column] = np.linspace(before, after, gap_size + 2)[1:-1]
            result.loc[start:end, "tracking_status"] = "interpolated"
    return result


def _overlay(frame: np.ndarray, row: pd.Series, trail: list[tuple[int, int]]) -> np.ndarray:
    image = frame.copy()
    if pd.notna(row.x_px):
        x, y = int(row.x_px), int(row.y_px)
        w, h = int(row.bbox_width_px), int(row.bbox_height_px)
        color = (0, 200, 0) if row.tracking_status == "detected" else (0, 180, 255)
        cv2.rectangle(image, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), color, 2)
        cv2.circle(image, (x, y), 4, color, -1)
        trail.append((x, y))
    if len(trail) > 1:
        cv2.polylines(image, [np.asarray(trail, dtype=np.int32)], False, (255, 100, 0), 2)
    cv2.putText(image, str(row.tracking_status), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
    return image


def _write_overlay_video(path: str | Path, data: pd.DataFrame, metadata: VideoMetadata, output_path: Path) -> None:
    """Render after interpolation so the output faithfully displays filled gaps."""
    capture = cv2.VideoCapture(str(path))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), metadata.fps, (metadata.width, metadata.height))
    trail: list[tuple[int, int]] = []
    try:
        for _, row in data.iterrows():
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(_overlay(frame, row, trail))
    finally:
        capture.release()
        writer.release()


def track_video(path: str | Path, detector: YOLOBobDetector, output_path: str | Path | None = None,
                progress: Callable[[float], None] | None = None, max_gap_frames: int = 10) -> ProcessingResult:
    metadata = validate_video(path)
    capture = cv2.VideoCapture(str(path))
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, previous = [], None
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            candidates = detector.detect(frame)
            detection = detector.select_candidate(candidates, previous)
            if detection:
                previous = detection
                row = {"frame_number": frame_number, "timestamp_s": frame_number / metadata.fps,
                       "x_px": detection.x, "y_px": detection.y, "bbox_width_px": detection.width,
                       "bbox_height_px": detection.height, "confidence": detection.confidence, "tracking_status": "detected"}
            else:
                row = {"frame_number": frame_number, "timestamp_s": frame_number / metadata.fps,
                       "x_px": np.nan, "y_px": np.nan, "bbox_width_px": np.nan, "bbox_height_px": np.nan,
                       "confidence": np.nan, "tracking_status": "missing"}
            rows.append(row)
            frame_number += 1
            if progress:
                progress(min(frame_number / metadata.frame_count, 1.0))
    finally:
        capture.release()
    raw = pd.DataFrame(rows)
    data = interpolate_missing(raw, max_gap_frames)
    if output_path:
        _write_overlay_video(path, data, metadata, output_path)
    warnings = []
    if raw.empty or raw.tracking_status.eq("detected").sum() == 0:
        warnings.append("No bob was detected. Try a custom-trained bob model, a lower confidence threshold, or clearer footage.")
    elif data.tracking_status.eq("interpolated").mean() > .1:
        warnings.append("More than 10% of frames were interpolated; treat derived measurements cautiously.")
    if data.tracking_status.eq("missing").any():
        warnings.append("Some tracking gaps exceeded the interpolation limit and remain missing; analysis excludes them.")
    if raw.confidence.dropna().mean() < .45 if raw.confidence.notna().any() else False:
        warnings.append("Mean detection confidence is low. Check bounding boxes before using the analysis in a report.")
    return ProcessingResult(data=data, metadata=metadata, processed_video=Path(output_path) if output_path else None, warnings=warnings)
