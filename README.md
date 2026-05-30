# AutoDetect-Wildlife
**CS-280 Robotics Final — North Idaho College, 2026**

Real-time wildlife detection system with two deployment targets: a laptop/desktop path using YOLOv11s and a Raspberry Pi 5 path using a custom INT8-quantized ONNX model — no PyTorch required on the Pi.

---

## Overview

Wildlife near roads and trails often goes undetected until it's too late. This project builds a low-cost, fully **offline** wildlife detector that runs on a Raspberry Pi 5, purpose-trained on deer, elk, turkey, and moose from real-world datasets.

| | Laptop / Desktop | Raspberry Pi 5 |
|---|---|---|
| **Model** | YOLOv11s `.pt` | YOLOv11n INT8 ONNX |
| **Runtime** | PyTorch / Ultralytics | ONNX Runtime (CPU only) |
| **Camera** | USB webcam | Pi Camera Module |
| **Input size** | 640×480 | 864×480 (16:9) |
| **Output** | OpenCV window or MJPEG stream | OpenCV window or terminal |
| **FPS** | 60–80 FPS | ~3 FPS (INT8, CPU) |

---

## Training Pipeline

1. **Dataset merging** — Roboflow dataset (deer, elk, turkey, moose, ~3,000 images) + LILA.science Caltech Camera Traps subset merged into a unified structure
2. **Base training** (`train_wildlife.py`) — YOLOv11s for laptop, YOLOv11n for Pi base weights
3. **Pi-optimized export** (`train_rpi.py`) — fine-tunes YOLOv11n at 864×480, exports FP32 ONNX, then quantizes to INT8 → `best_int8.onnx` copied to Pi, no PyTorch needed

---

## Running

### Install dependencies
```bash
pip install --user -r requirements.txt
```
On Raspberry Pi: swap `opencv-python` for `opencv-python-headless` and add `onnxruntime picamera2`.

### Laptop — live webcam detection
```bash
python AutoWildLife.py                   # OpenCV window
python AutoWildLife.py --headless        # terminal output + MJPEG stream on :8080
```

### Raspberry Pi — ONNX inference (no PyTorch)
```bash
python detect_wildlife.py                # Pi camera live feed
python detect_wildlife.py --headless     # terminal output only
python detect_wildlife.py --source video.mp4
python detect_wildlife.py --source image.jpg
```

Headless terminal output example:
```
[14:32:07] Deer 91%  box=(120,88,430,390)  3.2FPS  312ms
[14:32:08] Elk  78%  box=(50,200,310,480)  3.1FPS  318ms
```

### Training (desktop with GPU)
```bash
python train_wildlife.py          # train both laptop and Pi models
python train_wildlife.py --rpi-only   # Pi model only (faster)
python train_rpi.py               # INT8 ONNX export for Pi (run after above)
```

Press `q` to quit any OpenCV window. `Ctrl+C` to stop headless mode.

---

## Detection Features

- **Close-animal warning** — bounding box turns red when a detected animal exceeds 35% of frame height, indicating a likely collision risk (threshold tunable via `WARNING_SIZE_THRESHOLD`)
- **Class filtering** — COCO pretrained mode restricts output to road-relevant animals; custom model mode trusts all class outputs directly
- **Smoothed FPS counter** — rolling 20-frame average keeps the display readable

---

## Architecture

**Laptop path** (`AutoWildLife.py`): loads `.pt` via Ultralytics, uses CUDA if available. Supports `--headless` to skip display and serve live MJPEG over HTTP on port 8080.

**Raspberry Pi path** (`detect_wildlife.py`): loads INT8 ONNX via `onnxruntime` — zero PyTorch dependency. Uses `picamera2` for the Pi camera, falls back to `cv2.VideoCapture` for USB cameras or video files. Custom letterbox preprocessing and YOLO output postprocessing with NMS.

---

## Tech Stack

`Python` `YOLOv11` `ONNX Runtime` `OpenCV` `Ultralytics` `Raspberry Pi 5` `INT8 Quantization`
