"""End-to-end checks using two generated videos and a deterministic bob detector."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.detection.yolo_detector import Detection
from app.tracking.video_tracker import track_video, validate_video


class SyntheticBobDetector:
    """Detect the green synthetic bob; selected frames intentionally simulate occlusion."""

    def __init__(self, misses: set[int] | None = None):
        self.frame = 0
        self.misses = misses or set()

    def detect(self, image: np.ndarray) -> list[Detection]:
        frame_number = self.frame
        self.frame += 1
        if frame_number in self.misses:
            return []
        mask = cv2.inRange(image, np.array([0, 180, 0], dtype=np.uint8), np.array([80, 255, 80], dtype=np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        x, y, width, height = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return [Detection(x + width / 2, y + height / 2, width, height, .99)]

    @staticmethod
    def select_candidate(candidates, previous):
        return candidates[0] if candidates else None


def make_pendulum_video(path: Path, phase: float) -> None:
    width, height, frames, fps = 160, 120, 24, 12
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((height, width, 3), 25, dtype=np.uint8)
        x = int(80 + 35 * np.sin(index / 4 + phase))
        y = int(70 + 8 * np.cos(index / 4 + phase))
        cv2.circle(image, (x, y), 8, (0, 255, 0), -1)
        writer.write(image)
    writer.release()


def test_two_videos_complete_tracking_pipeline(tmp_path: Path):
    video_one, video_two = tmp_path / "pendulum_one.mp4", tmp_path / "pendulum_two.mp4"
    make_pendulum_video(video_one, 0.0)
    make_pendulum_video(video_two, 1.1)
    for video, misses in ((video_one, {5, 6}), (video_two, {12})):
        metadata = validate_video(video)
        output = video.with_name(f"{video.stem}_tracked.mp4")
        result = track_video(video, SyntheticBobDetector(misses), output)
        assert metadata.frame_count == 24
        assert len(result.data) == 24
        assert result.data.tracking_status.eq("detected").sum() == 24 - len(misses)
        assert result.data.tracking_status.eq("interpolated").sum() == len(misses)
        assert result.data[["x_px", "y_px"]].notna().all().all()
        assert result.data.loc[result.data.tracking_status.eq("interpolated"), "raw_x_px"].isna().all()
        assert output.exists() and output.stat().st_size > 0


def test_long_or_unbounded_tracking_gaps_are_never_fabricated(tmp_path: Path):
    video = tmp_path / "long_gap.mp4"
    make_pendulum_video(video, 0.0)
    result = track_video(video, SyntheticBobDetector({0, 1, 8, 9, 10, 11, 12}), max_gap_frames=3)
    assert result.data.tracking_status.eq("interpolated").sum() == 0
    assert result.data.tracking_status.eq("missing").sum() == 7
    assert result.data.loc[result.data.tracking_status.eq("missing"), "x_px"].isna().all()
