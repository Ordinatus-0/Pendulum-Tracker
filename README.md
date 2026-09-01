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

## Architecture

* `app/detection/yolo_detector.py` loads YOLOv8 and turns its boxes into `Detection` records. Supply custom weights and optionally a bob class ID in the UI; this is recommended because COCO's pretrained classes do not include pendulum bobs.
* `app/tracking/video_tracker.py` validates video input, invokes detection per frame, selects the closest high-confidence candidate to the prior bob position, records all required measurements, linearly interpolates missing positions while marking them as `interpolated`, and writes an annotated MP4.
* `app/analysis/motion.py` converts pixels to pivot-relative coordinates (right/up convention), smooths only a copy using Savitzky–Golay, calculates `r` and unwrapped `theta`, estimates amplitude decay, and fits the optional model `r = A exp(-k theta)` using nonlinear least squares.
* `app/visualization` produces separate x–time, y–time, x–y, r–theta, and spiral-fit charts. `app/export` writes CSV, JSON, PNG, and SVG assets.

## Analysis and reliability notes

The pivot is deliberately user-configurable: a camera cannot reliably infer a pivot without calibration and geometry assumptions. Raw pixel columns (`x_px`, `y_px`) are kept intact. Relative coordinates use the pivot as `(0,0)`, with +x right and +y upward. The application labels each frame `detected`, `missing`, or `interpolated`; it never treats filled values as original measurements.

The spiral fit is reported only as reliable when its R² is at least 0.50 and its fitted decay is non-negative. It is an exploratory model, not generally the physical path of an ideal pendulum. Interpret damping and displacement in pixel units unless a camera calibration is supplied. Perspective, camera tilt, occlusions, motion blur, and low YOLO confidence can bias results.
