import cv2
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Callable, Optional, cast

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detectors.yolo11 import YOLO11Detector  # noqa: F401 (kept for future use)
from src.trackers.yolo_bytetrack import YOLOByteTrackWrapper
from src.speed.speed import SpeedEstimator
from src.metrics.counting import LineCounter
from src.utils import denorm_line, draw_counting_line, draw_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='path to video file')
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/demo')
    ap.add_argument('--device', default=None)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--min-speed', type=float, default=1.0,
                    help='minimum speed (km/h) to consider an object as moving for visualization')
    ap.add_argument('--out-name', type=str, default=None,
                    help='explicit output filename (no path). .mp4 added if missing')
    ap.add_argument('--yolo-weights', type=str, default=None,
                    help='YOLO weights path or alias (e.g., yolo11l.pt). '
                         'If omitted, uses config.models.yolo11l')
    args = ap.parse_args()

    # Resolve video path
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        repo_root = Path(__file__).resolve().parents[1]
        candidate = (repo_root / args.video).resolve()
        if candidate.exists():
            video_path = candidate
    args.video = str(video_path)

    cfg = yaml.safe_load(open(args.config, 'r'))
    fps = cfg.get('fps', 30)
    ppm = cfg.get('pixel_per_meter', 10.0)
    line_norm = cfg.get('counting_line', [0.2, 0.8, 0.8, 0.8])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        attempted = args.video
        cwd = Path.cwd()
        raise AssertionError(
            f"Cannot open video: {attempted}\n"
            f"CWD: {cwd}\n"
            f"Try an absolute path, e.g.\n"
            f"  python ./scripts/run_yolo11_pipeline.py --video \"C:/full/path/video.mp4\" --config ./configs/config.yaml"
        )
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = None
    except Exception:
        total_frames = None
    cap.release()

    line_px = denorm_line(line_norm, W, H)

    os.makedirs(args.out, exist_ok=True)

    models_cfg = cfg.get('models', {}) or {}
    # Pick YOLO weights: CLI overrides config
    yolo_weights = args.yolo_weights or models_cfg.get('yolo11l', 'yolo11l.pt')

    # Build output filename
    video_stem = Path(args.video).stem
    def _flt_name(x):
        try:
            s = f"{float(x):.2f}"
        except Exception:
            s = str(x)
        return s.replace('.', 'p')
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.out_name:
        out_fname = args.out_name if args.out_name.lower().endswith('.mp4') else args.out_name + '.mp4'
    else:
        model_name = Path(str(yolo_weights)).stem
        out_fname = f"{video_stem}_{model_name}_conf{_flt_name(args.conf)}_min{_flt_name(args.min_speed)}_{ts}.mp4"
    out_path = str(Path(args.out) / out_fname)

    fourcc_fn: Optional[Callable[..., int]] = cast(Optional[Callable[..., int]], getattr(cv2, 'VideoWriter_fourcc', None))
    if fourcc_fn is None:
        def _fourcc_fallback(a: str, b: str, c: str, d: str) -> int:
            return (ord(a) & 0xff) | ((ord(b) & 0xff) << 8) | ((ord(c) & 0xff) << 16) | ((ord(d) & 0xff) << 24)
        fourcc_fn = _fourcc_fallback
    fourcc = fourcc_fn(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    tracker = YOLOByteTrackWrapper(model_path=yolo_weights, conf=args.conf, device=args.device)
    speed = SpeedEstimator(fps=fps, pixel_per_meter=ppm)
    counter = LineCounter((line_px[0], line_px[1]), (line_px[2], line_px[3]))

    processed_frames = 0
    print_interval = 10

    for frame, tracks in tracker.iter_tracks(args.video):
        speed_kmh = speed.update(tracks)
        left_to_right, right_to_left = counter.update(tracks)

        draw_counting_line(frame, line_px, color=(0, 255, 0))
        for t in tracks:
            tid = t['id']
            v = speed_kmh.get(tid, 0.0)
            if v < args.min_speed:
                continue
            x1, y1, x2, y2 = map(int, t['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            draw_label(frame, f'ID {tid} | {v:.1f} km/h', x1, max(20, y1 - 10))

        draw_label(frame, f'Count L->R: {left_to_right}', 10, 30)
        draw_label(frame, f'Count R->L: {right_to_left}', 10, 60)

        writer.write(frame)

        processed_frames += 1
        if total_frames:
            pct = processed_frames / total_frames * 100.0
            update_every = max(1, total_frames // 100)
            if (processed_frames % update_every == 0) or (processed_frames % print_interval == 0) or (processed_frames == total_frames):
                print(f'Processing: {processed_frames}/{total_frames} ({pct:.1f}%)', end='\r', flush=True)
        else:
            if processed_frames % print_interval == 0:
                print(f'Processed frames: {processed_frames}', end='\r', flush=True)

    print()
    writer.release()
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
