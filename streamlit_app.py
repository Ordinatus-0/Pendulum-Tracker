"""Local Streamlit entry point for Pendulum Tracker."""
from __future__ import annotations

import tempfile
from io import BytesIO
from pathlib import Path

import streamlit as st

from app.analysis.motion import fit_spiral, motion_summary, smooth_coordinates, transform_coordinates
from app.detection.yolo_detector import YOLOBobDetector
from app.export.files import save_figures, write_csv, write_summary
from app.tracking.video_tracker import validate_video, track_video
from app.visualization.plots import motion_figures, spiral_figure

st.set_page_config(page_title="Pendulum Tracker", layout="wide")
st.title("Pendulum Tracker")
st.caption("YOLOv8-based bob tracking and report-ready motion analysis. Pixel coordinates are not calibrated physical lengths.")

uploaded = st.file_uploader("Upload Video", type=["mp4", "mov", "avi", "mkv"])
if uploaded:
    suffix = Path(uploaded.name).suffix
    video_path = Path(tempfile.gettempdir()) / f"pendulum_upload{suffix}"
    video_path.write_bytes(uploaded.getvalue())
    try:
        metadata = validate_video(video_path)
    except ValueError as exc:
        st.error(str(exc)); st.stop()
    st.video(uploaded)
    columns = st.columns(5)
    for col, label, value in zip(columns, ["FPS", "Frames", "Duration", "Width", "Height"], [metadata.fps, metadata.frame_count, f"{metadata.duration_seconds:.2f} s", metadata.width, metadata.height]):
        col.metric(label, value)
    with st.sidebar:
        st.header("Detection settings")
        weights = st.text_input("YOLO weights", "yolov8n.pt", help="Use a custom-trained bob model for best results.")
        confidence = st.slider("Minimum confidence", .05, .95, .25, .05)
        class_text = st.text_input("Custom bob class ID (optional)", "")
        smooth_window = st.slider("Smoothing window (frames)", 3, 101, 21, 2)
    if st.button("Run Detection & Tracking", type="primary"):
        progress = st.progress(0, text="Loading YOLO and processing frames…")
        detector = YOLOBobDetector(weights, confidence, int(class_text) if class_text.strip() else None)
        try:
            result = track_video(video_path, detector, Path(tempfile.gettempdir()) / "pendulum_tracked.mp4", lambda p: progress.progress(p, text=f"Processing {p:.0%}"))
        except Exception as exc:
            st.exception(exc); st.stop()
        st.session_state.result = result
        progress.empty()
    if "result" in st.session_state:
        result = st.session_state.result
        st.subheader("Tracking result")
        for warning in result.warnings: st.warning(warning)
        if result.processed_video and result.processed_video.exists(): st.video(result.processed_video.read_bytes())
        pivot_cols = st.columns(2)
        pivot_x = pivot_cols[0].number_input("Pivot x (px)", value=float(metadata.width / 2))
        pivot_y = pivot_cols[1].number_input("Pivot y (px)", value=0.0, help="Set manually; automatic pivot estimation is intentionally not assumed reliable.")
        if st.button("Analyse Motion"):
            data = smooth_coordinates(transform_coordinates(result.data, pivot_x, pivot_y), smooth_window)
            st.session_state.analysis_data = data
            st.session_state.fit = fit_spiral(data)
        if "analysis_data" in st.session_state:
            data, fit = st.session_state.analysis_data, st.session_state.fit
            summary = motion_summary(data)
            st.subheader("Lab-report summary")
            st.json({**summary, "spiral_A": fit.get("A"), "spiral_k": fit.get("k"), "spiral_r_squared": fit.get("r_squared"), "spiral_reliable": fit["reliable"]})
            if not fit["reliable"]: st.warning(f"Spiral fit is unreliable: {fit.get('reason', 'insufficient agreement with data')}")
            use_smooth = st.toggle("Show smoothed measured trajectories", value=True)
            figures = motion_figures(data, use_smooth); figures["spiral_fit"] = spiral_figure(fit)
            st.subheader("Plots")
            for figure in figures.values(): st.pyplot(figure)
            export_dir = Path(tempfile.gettempdir()) / "pendulum_exports"
            write_csv(data, export_dir / "tracking_data.csv")
            write_summary(summary, fit, result.warnings + ["Perspective, camera angle, occlusion, and pixel calibration can bias results."], export_dir / "summary.json")
            save_figures(figures, export_dir)
            st.download_button("Export Data (CSV)", (export_dir / "tracking_data.csv").read_bytes(), "pendulum_tracking.csv", "text/csv")
            st.download_button("Export Summary (JSON)", (export_dir / "summary.json").read_bytes(), "pendulum_summary.json", "application/json")
            figure_name = st.selectbox("Figure to export", list(figures))
            png = BytesIO(); figures[figure_name].savefig(png, format="png", dpi=200, bbox_inches="tight")
            svg = BytesIO(); figures[figure_name].savefig(svg, format="svg", bbox_inches="tight")
            st.download_button("Export Figure (PNG)", png.getvalue(), f"{figure_name}.png", "image/png")
            st.download_button("Export Figure (SVG)", svg.getvalue(), f"{figure_name}.svg", "image/svg+xml")
            if result.processed_video and result.processed_video.exists():
                st.download_button("Export Processed Video", result.processed_video.read_bytes(), "pendulum_tracked.mp4", "video/mp4")
else:
    st.info("Upload a video to begin. For reliable automatic detection, supply custom YOLOv8 weights trained on your pendulum bob.")
