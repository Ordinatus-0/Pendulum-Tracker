# Pendulum Tracker

A local Streamlit application that tracks a pendulum bob in an uploaded video with YOLOv8, preserves frame-level measurement provenance, and creates physics-lab-report plots and exports.

## Setup and run

Requires Python 3.10+ and a working OpenCV video backend.

```bash
git clone <repository-url>
cd Pendulum-Tracker
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit, upload an MP4/MOV/AVI/MKV video, set the detection options, then click **Run Detection & Tracking** and **Analyse Motion**. Run tests with `pytest`.

## Pretrained single-pendulum weights

No third-party binary checkpoint is committed to this repository. A search found a publicly hosted YOLOv8 checkpoint for a **single pendulum** (`danbai_best.pt`) in the `qiaoyuzheng0804-create/pendulum_web` repository. Its repository does not provide a licence file, so this project supplies a checksum-verified downloader rather than redistributing its binary. Download it only after reviewing the upstream repository and its terms:

```bash
python scripts/download_pretrained_bob_weights.py
```

Then enter `models/danbai_best.pt` as **YOLO weights** and `0` as **Custom bob class ID**. This is an externally trained single-pendulum checkpoint, not a model trained or validated by this project; verify its detections on your own footage before using results. The included `bob_dataset.yaml` remains the recommended route for reproducible, appropriately licensed weights.

The automated test suite also creates and processes two independent, short synthetic MP4 pendulum videos. It verifies their metadata, successful frame-by-frame tracking, explicit interpolation during simulated occlusions, and annotated-video generation without requiring downloaded YOLO weights.

## Architecture

* `app/detection/yolo_detector.py` loads the supplied Ultralytics `yolov8n.pt` weights by default (they download on first use) and turns its boxes into `Detection` records. `bob_dataset.yaml` includes a one-class custom label (`0: bob`) for training custom weights; use class ID `0` with those weights because COCO's pretrained classes do not include pendulum bobs.
* `app/tracking/video_tracker.py` validates video input, invokes detection per frame, selects the closest high-confidence candidate to the prior bob position, records all required measurements, linearly interpolates missing positions while marking them as `interpolated`, and writes an annotated MP4.
* `app/analysis/motion.py` converts pixels to pivot-relative coordinates (right/up convention), smooths only a copy using Savitzky–Golay, calculates `r` and unwrapped `theta`, estimates amplitude decay, fits the exponential spiral `r = A exp(-k theta)`, and compares polynomial degrees 0–6 and sine/cosine curves (one to three harmonics) using AICc.
* `app/visualization` produces separate x–time, y–time, x–y, r–theta, best-model, and spiral-fit charts. `app/export` writes CSV tables for tracking data, scored candidate models, and both fitted curves, plus JSON, PNG, and SVG assets.

## Analysis and reliability notes

The pivot is deliberately user-configurable: a camera cannot reliably infer a pivot without calibration and geometry assumptions. Raw pixel columns (`x_px`, `y_px`) are kept intact. Relative coordinates use the pivot as `(0,0)`, with +x right and +y upward. The application labels each frame `detected`, `missing`, or `interpolated`; it never treats filled values as original measurements.

The application selects the best-supported radial model by AICc, so it does not assume a pendulum follows a spiral; the exponential spiral is still reported, plotted, and exported for direct comparison. Fits with R² below 0.50 are labelled unreliable, and spiral fits additionally require non-negative decay. These are exploratory image-space models, not a physical proof of pendulum dynamics. Interpret damping and displacement in pixel units unless a camera calibration is supplied. Perspective, camera tilt, occlusions, motion blur, and low YOLO confidence can bias results.
