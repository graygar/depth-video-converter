# Depth Video Converter

Convert any MP4 / MOV video into a grayscale **depth-map video** using
[Depth Anything V2](https://huggingface.co/depth-anything), with a simple
[Gradio](https://gradio.app) web UI. Everything runs **locally** — no video ever
leaves your machine.

## Features

- **MP4 / MOV upload** (also accepts M4V, AVI, MKV, WebM)
- **Depth Anything V2** — Small / Base / Large model selection
- **Automatic hardware acceleration**
  - NVIDIA CUDA on Windows/Linux when available
  - Apple Silicon (MPS) on macOS when available
  - CPU fallback everywhere else
- **Output resolution** control (original / 1080p / 720p / 480p)
- **Invert depth** (near = white by default; invert for near = black)
- **Temporal smoothing** to reduce frame-to-frame flicker
- **Keeps the original audio** (optional, via ffmpeg)
- Exports a widely compatible **H.264 MP4** (`yuv420p`, faststart)

## Requirements

- Python **3.10 – 3.12**
- ~2 GB of disk for PyTorch + model weights
  (Small ≈ 100 MB, Base ≈ 390 MB, Large ≈ 1.3 GB — downloaded automatically on
  first use and cached in `~/.cache/huggingface`)
- ffmpeg is **not** required separately — a bundled binary from
  `imageio-ffmpeg` is used automatically if ffmpeg isn't on your PATH.

## Installation

First clone the repository:

```bash
git clone https://github.com/graygar/depth-video-converter.git
cd depth-video-converter
```

### Windows

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate

REM With an NVIDIA GPU — install the CUDA build of PyTorch first:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

> No NVIDIA GPU? Skip the CUDA line — `requirements.txt` installs the CPU
> build and the app falls back to CPU automatically.

### macOS (Apple Silicon or Intel)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default PyTorch wheel already includes Apple Silicon (MPS) acceleration —
nothing extra to install.

## Launch

```bash
python app.py
```

Then open **http://127.0.0.1:7860** (a browser tab opens automatically).

On Windows use `python app.py` from the activated `.venv` as well.

## Usage

1. Drop an MP4/MOV into **Source video** (or record from the webcam).
2. Pick a **Model** — Small is the best speed/quality trade-off; Large is
   noticeably better but several times slower.
3. Pick an **Output resolution** — inference cost is the same regardless
   (frames are resized internally by the model), this only affects the
   output file size.
4. Toggle **Invert depth**, **Keep original audio**, **Reduce flicker** as
   needed.
5. Click **Convert video**. Progress appears on the button and in the
   status box; the finished video shows on the right and is also saved to
   the `outputs/` folder next to `app.py`.

## How it works

Each frame is decoded with OpenCV, run through Depth Anything V2, and the
predicted relative depth is normalized to 8-bit grayscale (near = white).
Flicker reduction blends each depth map with the previous frame (EMA) and
smooths the normalization range across frames, so global brightness doesn't
pump between frames. The frames are written to a temporary MP4, then ffmpeg
re-encodes to H.264/`yuv420p` and muxes in the source audio track.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| First conversion is slow to start | The model is being downloaded (one-time); watch the status box. |
| `CUDA available: False` on Windows with an NVIDIA GPU | You installed the CPU wheel. Re-run the `pip install torch ... cu124` line inside the venv, then reinstall requirements. |
| Out-of-memory on GPU | Use the Small model, or close other GPU apps. |
| Output has no audio | The source had no audio track, or "Keep original audio" was off. |
| MOV won't decode | Rarely, exotic codecs aren't supported by OpenCV's bundled FFmpeg — re-export the clip as H.264 MP4 first. |

## License

[MIT](LICENSE). Model weights are downloaded from Hugging Face at runtime and
carry their own licenses (Depth Anything V2 Small is Apache-2.0; Base and
Large are CC-BY-NC-4.0 — check before commercial use).
