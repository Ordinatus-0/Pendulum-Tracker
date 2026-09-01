"""Download the externally hosted single-pendulum YOLOv8 checkpoint.

The checkpoint is intentionally not committed to this repository: it is a
third-party 6.3 MB binary with no licence file in its upstream repository.
This script keeps provenance explicit and verifies the exact downloaded file.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen


SOURCE_URL = (
    "https://raw.githubusercontent.com/qiaoyuzheng0804-create/pendulum_web/"
    "main/models/danbai_best.pt"
)
SHA256 = "f62c7ff8b7e8c4fe089c758cf63cbcdf4b00bded212ab99df238c0d689dba493"


def download(destination: Path) -> Path:
    """Download and checksum the discovered single-pendulum checkpoint."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != SHA256:
        raise ValueError(f"Checksum mismatch: expected {SHA256}, got {digest}.")
    destination.write_bytes(payload)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a verified third-party single-pendulum YOLOv8 weight file.")
    parser.add_argument("--output", type=Path, default=Path("models/danbai_best.pt"))
    print(download(parser.parse_args().output))
