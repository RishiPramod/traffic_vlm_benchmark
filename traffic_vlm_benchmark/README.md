# Traffic VLM Benchmark

A minimal, modular benchmark to compare detectors and vision-language models for traffic analysis:
- Detection & Tracking: YOLO11-L (working), RT-DETR-L (stub), GroundingDINO (text-guided, stub)
- Reasoning VLMs: InternVL2, VideoLLaMA-2, CoT-VLM4Tar (stubs)
- Segmentation: SAM 2 (stub)
- Utilities: Counting lines, speed estimation, MOTA/IDF1 via motmetrics.

## Quickstart

```bash
# Python 3.10+ recommended
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip

# Core deps
pip install ultralytics supervision opencv-python numpy pandas tqdm matplotlib
pip install motmetrics
# Optional/advanced (comment out if you don't need them right away)
# pip install transformers accelerate sentencepiece bitsandbytes
# pip install groundingdino-py
# pip install sam2

# Run YOLO11-L tracking + counting + speed on your videos
python scripts/run_yolo11_pipeline.py   --video "/mnt/data/tracking_yolo11l_bytetrack (2).mp4"   --config configs/config.yaml   --out runs/demo
```

## Structure
- `src/detectors/yolo11.py`: YOLO11-L detection + ByteTrack tracking via Ultralytics
- `src/speed/speed.py`: speed estimation (pixel->meter scale + FPS)
- `src/metrics/*`: helpers for MOTA / IDF1 using motmetrics
- `src/vl/*`: stubs for GroundingDINO, CLIP, InternVL2, VideoLLaMA-2
- `scripts/run_yolo11_pipeline.py`: end-to-end demo (boxes + IDs + counts + speed)
- `scripts/evaluate.py`: skeleton for benchmarking multiple models

## Notes
- For accurate speed, provide either homography or a pixel-per-meter scale with camera FPS.
- Stubs are clean entry points; fill `TODO` to activate extra models when ready.
- Keep results under `runs/` for reproducibility.
