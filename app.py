"""
Depth Video Converter
=====================
Convert any MP4/MOV video into a grayscale depth-map video using
Depth Anything V2, with a simple Gradio web UI.

Runs fully locally:
  * NVIDIA GPU (CUDA)  -> used automatically on Windows/Linux when available
  * Apple Silicon (MPS) -> used automatically on macOS when available
  * CPU                 -> fallback everywhere

Launch:  python app.py   (then open http://127.0.0.1:7860)
"""

import os

# Allow PyTorch to fall back to CPU for any op MPS doesn't support.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_CHOICES = {
    "Small (fast, recommended)": "depth-anything/Depth-Anything-V2-Small-hf",
    "Base (better quality)": "depth-anything/Depth-Anything-V2-Base-hf",
    "Large (best quality, slow)": "depth-anything/Depth-Anything-V2-Large-hf",
}

RESOLUTION_CHOICES = {
    "Original resolution": None,
    "1080p (long edge 1920)": 1920,
    "720p (long edge 1280)": 1280,
    "480p (long edge 854)": 854,
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# How strongly the previous frame's depth is blended into the current one
# when "Reduce flicker" is enabled. 0 = off, higher = smoother but more ghosting.
TEMPORAL_BLEND = 0.45
# EMA factor for the normalization range (percentile lo/hi) across frames.
RANGE_EMA = 0.90


# --------------------------------------------------------------------------- #
# Device & model handling
# --------------------------------------------------------------------------- #

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = pick_device()

_model_cache: dict[str, tuple] = {}
_model_lock = threading.Lock()


def load_model(model_id: str):
    """Load (and cache) a Depth Anything V2 model + processor."""
    with _model_lock:
        if model_id in _model_cache:
            return _model_cache[model_id]
        try:
            processor = AutoImageProcessor.from_pretrained(model_id, use_fast=True)
        except Exception:
            processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)
        model.to(DEVICE).eval()
        _model_cache[model_id] = (processor, model)
        return processor, model


# --------------------------------------------------------------------------- #
# ffmpeg helpers
# --------------------------------------------------------------------------- #

def find_ffmpeg() -> str | None:
    """Prefer a system ffmpeg; fall back to the binary bundled with imageio-ffmpeg."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def encode_final(ffmpeg: str, raw_video: str, source_video: str,
                 out_path: str, keep_audio: bool) -> bool:
    """Re-encode the raw OpenCV output to H.264 (+AAC audio from the source)."""
    cmd = [ffmpeg, "-y", "-i", raw_video]
    if keep_audio:
        cmd += ["-i", source_video]
    cmd += ["-map", "0:v:0"]
    if keep_audio:
        cmd += ["-map", "1:a:0?"]  # audio is optional; silent sources still work
    cmd += [
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
    ]
    if keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += [out_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0 and Path(out_path).exists()


# --------------------------------------------------------------------------- #
# Depth inference
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def infer_depth_batch(processor, model, frames_rgb: list[np.ndarray],
                      out_size: tuple[int, int]) -> np.ndarray:
    """Run depth estimation on a batch of RGB frames.

    Returns a float32 array of shape (B, out_h, out_w) with raw (relative)
    depth values — larger means closer to the camera.
    """
    images = [Image.fromarray(f) for f in frames_rgb]
    inputs = processor(images=images, return_tensors="pt").to(DEVICE)

    if DEVICE == "cuda":
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            predicted = model(**inputs).predicted_depth
    else:
        predicted = model(**inputs).predicted_depth

    predicted = F.interpolate(
        predicted.unsqueeze(1).float(),
        size=out_size,  # (h, w)
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)
    return predicted.cpu().numpy()


def compute_output_size(width: int, height: int, long_edge: int | None) -> tuple[int, int]:
    """Return (out_w, out_h), scaled to the requested long edge, forced even."""
    if long_edge is not None and max(width, height) > long_edge:
        scale = long_edge / max(width, height)
        width = int(round(width * scale))
        height = int(round(height * scale))
    # H.264 requires even dimensions
    return max(2, width - width % 2), max(2, height - height % 2)


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #

def convert_video(video_path: str | None,
                  model_label: str,
                  resolution_label: str,
                  invert: bool,
                  keep_audio: bool,
                  reduce_flicker: bool,
                  progress=gr.Progress()):
    log: list[str] = []

    def status(msg: str) -> str:
        log.append(msg)
        return "\n".join(log)

    if not video_path:
        raise gr.Error("Please upload a video first.")

    src = Path(video_path)
    if src.suffix.lower() not in (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"):
        raise gr.Error(f"Unsupported file type: {src.suffix}")

    device_names = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple Silicon (MPS)", "cpu": "CPU"}
    yield None, status(f"Device: {device_names[DEVICE]}")

    model_id = MODEL_CHOICES[model_label]
    yield None, status(f"Loading model {model_id} (first run downloads it)...")
    t0 = time.time()
    processor, model = load_model(model_id)
    yield None, status(f"Model ready in {time.time() - t0:.1f}s")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise gr.Error("Could not open the video. Is it a valid MP4/MOV file?")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_w, out_h = compute_output_size(in_w, in_h, RESOLUTION_CHOICES[resolution_label])
    yield None, status(f"Input {in_w}x{in_h} @ {fps:.2f} fps -> output {out_w}x{out_h}")

    tmp_dir = tempfile.mkdtemp(prefix="depthvid_")
    raw_path = str(Path(tmp_dir) / "depth_raw.mp4")
    writer = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise gr.Error("Could not create the output video writer.")

    batch_size = {"cuda": 8, "mps": 4, "cpu": 1}[DEVICE]

    # Temporal smoothing state
    prev_depth: np.ndarray | None = None
    ema_lo: float | None = None
    ema_hi: float | None = None

    def write_depth_frame(depth: np.ndarray):
        nonlocal prev_depth, ema_lo, ema_hi
        if reduce_flicker and prev_depth is not None:
            depth = TEMPORAL_BLEND * prev_depth + (1.0 - TEMPORAL_BLEND) * depth
        prev_depth = depth

        lo = float(np.percentile(depth, 1.0))
        hi = float(np.percentile(depth, 99.0))
        if reduce_flicker and ema_lo is not None:
            lo = RANGE_EMA * ema_lo + (1.0 - RANGE_EMA) * lo
            hi = RANGE_EMA * ema_hi + (1.0 - RANGE_EMA) * hi
        ema_lo, ema_hi = lo, hi

        norm = np.clip((depth - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        if invert:
            norm = 1.0 - norm  # near = black instead of near = white
        gray = (norm * 255.0).astype(np.uint8)
        writer.write(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    t0 = time.time()
    done = 0
    buffer: list[np.ndarray] = []
    last_update = 0.0

    def flush():
        nonlocal done, last_update
        if not buffer:
            return None
        depths = infer_depth_batch(processor, model, buffer, (out_h, out_w))
        for d in depths:
            write_depth_frame(d)
        done += len(buffer)
        buffer.clear()
        now = time.time()
        if now - last_update > 0.5:
            last_update = now
            frac = done / total if total else 0.0
            rate = done / max(now - t0, 1e-6)
            progress(min(frac, 1.0), desc=f"Frame {done}/{total or '?'} ({rate:.1f} fps)")
        return None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        buffer.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(buffer) >= batch_size:
            flush()
    flush()
    cap.release()
    writer.release()

    if done == 0:
        raise gr.Error("No frames could be decoded from the video.")

    elapsed = time.time() - t0
    yield None, status(f"Processed {done} frames in {elapsed:.1f}s ({done / elapsed:.1f} fps)")

    # Final encode: H.264 + optional audio from the source
    final_path = str(OUTPUT_DIR / f"{src.stem}_depth_{int(time.time())}.mp4")
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        yield None, status("Encoding final MP4 (H.264" + (" + audio" if keep_audio else "") + ")...")
        if encode_final(ffmpeg, raw_path, str(src), final_path, keep_audio):
            os.remove(raw_path)
        else:
            shutil.move(raw_path, final_path)
            yield None, status("Warning: ffmpeg encode failed, kept the raw MP4 (no audio).")
    else:
        shutil.move(raw_path, final_path)
        yield None, status("Warning: ffmpeg not found, kept the raw MP4 (no audio, less compatible).")

    yield final_path, status(f"Done -> {final_path}")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

# Brand palette: orange accent on a near-black background with dark gray panels.
ACCENT = "#ff7a3b"
BACKGROUND = "#0d0d0d"
PANEL = "#2c2c2c"

LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo.svg"


def logo_html() -> str:
    """Inline the logo SVG (if present), scaled to a small header size."""
    if not LOGO_PATH.exists():
        return ""
    svg = LOGO_PATH.read_text(encoding="utf-8")
    # Drop the intrinsic pixel size; the viewBox keeps the aspect ratio.
    svg = re.sub(r'(<svg[^>]*?)\s+width="[^"]*"', r"\1", svg, count=1)
    svg = re.sub(r'(<svg[^>]*?)\s+height="[^"]*"', r"\1", svg, count=1)
    svg = svg.replace("<svg", '<svg style="height:32px;width:auto;display:block"', 1)
    return f'<div style="display:flex;align-items:center;padding:6px 0 2px;">{svg}</div>'

_orange = gr.themes.colors.Color(
    name="brand_orange",
    c50="#fff3ec", c100="#ffe5d6", c200="#ffcbad", c300="#ffab7d",
    c400="#ff9257", c500=ACCENT, c600="#f45f1c", c700="#cc4a12",
    c800="#a03a10", c900="#7d2f10", c950="#431806",
)


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Soft(primary_hue=_orange, neutral_hue=gr.themes.colors.zinc)
    # Force the same dark look in both light and dark browser modes.
    dual = {
        "body_background_fill": BACKGROUND,
        "background_fill_primary": PANEL,
        "background_fill_secondary": "#232323",
        "block_background_fill": PANEL,
        "panel_background_fill": PANEL,
        "input_background_fill": "#1f1f1f",
        "checkbox_background_color": "#1f1f1f",
        "checkbox_background_color_selected": ACCENT,
        "border_color_primary": "#3a3a3a",
        "block_border_color": "#3a3a3a",
        "input_border_color": "#3a3a3a",
        "body_text_color": "#e8e8e8",
        "body_text_color_subdued": "#9a9a9a",
        "block_title_text_color": "#e8e8e8",
        "block_label_text_color": "#b5b5b5",
        "block_info_text_color": "#9a9a9a",
        "block_label_background_fill": "#1f1f1f",
        "button_primary_background_fill": ACCENT,
        "button_primary_background_fill_hover": "#ff9257",
        "button_primary_text_color": "#1a1208",
        "button_secondary_background_fill": "#3a3a3a",
        "button_secondary_background_fill_hover": "#4a4a4a",
        "button_secondary_text_color": "#e8e8e8",
        "color_accent_soft": "#43281a",
        "loader_color": ACCENT,
        "slider_color": ACCENT,
        "link_text_color": ACCENT,
        "link_text_color_hover": "#ff9257",
        "table_even_background_fill": PANEL,
        "table_odd_background_fill": "#262626",
    }
    return theme.set(**dual, **{f"{k}_dark": v for k, v in dual.items()})


def build_ui() -> gr.Blocks:
    device_names = {"cuda": "NVIDIA GPU (CUDA)", "mps": "Apple Silicon (MPS)", "cpu": "CPU"}
    with gr.Blocks(title="Depth Video Converter", theme=build_theme()) as demo:
        if (logo := logo_html()):
            gr.HTML(logo)
        gr.Markdown(
            "# Depth Video Converter\n"
            f"Convert a video into a grayscale depth-map video with Depth Anything V2. "
            f"Running on **{device_names[DEVICE]}** — all processing stays local."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.Video(label="Source video (MP4 / MOV)", sources=["upload", "webcam"])
                model_dd = gr.Dropdown(
                    choices=list(MODEL_CHOICES),
                    value="Small (fast, recommended)",
                    label="Model",
                )
                res_dd = gr.Dropdown(
                    choices=list(RESOLUTION_CHOICES),
                    value="720p (long edge 1280)",
                    label="Output resolution",
                )
                invert_cb = gr.Checkbox(value=False, label="Invert depth (near = black)")
                audio_cb = gr.Checkbox(value=True, label="Keep original audio")
                flicker_cb = gr.Checkbox(value=True, label="Reduce flicker (temporal smoothing)")
                run_btn = gr.Button("Convert video", variant="primary")
            with gr.Column(scale=1):
                video_out = gr.Video(label="Depth output", interactive=False)
                status_tb = gr.Textbox(label="Conversion status", lines=10, interactive=False)

        run_btn.click(
            convert_video,
            inputs=[video_in, model_dd, res_dd, invert_cb, audio_cb, flicker_cb],
            outputs=[video_out, status_tb],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch(server_name="127.0.0.1", inbrowser=True)
