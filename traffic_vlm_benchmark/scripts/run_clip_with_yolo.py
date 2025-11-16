import cv2
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import Callable, Optional, cast

# repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.trackers.yolo_bytetrack import YOLOByteTrackWrapper
from src.speed.speed import SpeedEstimator
from src.metrics.counting import LineCounter
from src.utils import denorm_line, draw_counting_line, draw_label

# CLIP wrapper (already in your repo)
from src.vl.clip_utils import CLIPZeroShot


CANDIDATE_LABELS_DEFAULT = ["car", "bus", "truck", "motorcycle", "bicycle", "person", "auto-rickshaw", "van"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/clip')
    ap.add_argument('--device', default=None)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--min-speed', type=float, default=0.0)
    ap.add_argument('--yolo-weights', type=str, default='yolo11l.pt',
                    help='YOLO weights for tracking/detection (boxes come from YOLO)')
    ap.add_argument('--clip-model', type=str, default=None,
                    help='HF id for CLIP (defaults to config.models.clip)')
    ap.add_argument('--labels', type=str, default=None,
                    help='comma-separated labels (default: a small traffic set)')
    ap.add_argument('--out-name', type=str, default=None)
    args = ap.parse_args()

    # load config
    cfg = yaml.safe_load(open(args.config, 'r'))
    fps = cfg.get('fps', 30)
    ppm = cfg.get('pixel_per_meter', 10.0)
    line_norm = cfg.get('counting_line', [0.2, 0.8, 0.8, 0.8])

    models_cfg = cfg.get('models', {}) or {}
    clip_id = args.clip_model or models_cfg.get('clip', 'openai/clip-vit-base-patch32')

    # resolve video
    video_path = Path(args.video).expanduser()
    if not video_path.exists():
        repo_root = Path(__file__).resolve().parents[1]
        candidate = (repo_root / args.video).resolve()
        if candidate.exists():
            video_path = candidate
    args.video = str(video_path)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise AssertionError(f"Cannot open video: {args.video}")
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

    # output name
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
        out_fname = f"{video_stem}_yolo+clip_conf{_flt_name(args.conf)}_{ts}.mp4"
    out_path = str(Path(args.out) / out_fname)

    fourcc_fn: Optional[Callable[..., int]] = cast(Optional[Callable[..., int]], getattr(cv2, 'VideoWriter_fourcc', None))
    if fourcc_fn is None:
        def _fourcc_fallback(a: str, b: str, c: str, d: str) -> int:
            return (ord(a) & 0xff) | ((ord(b) & 0xff) << 8) | ((ord(c) & 0xff) << 16) | ((ord(d) & 0xff) << 24)
        fourcc_fn = _fourcc_fallback
    fourcc = fourcc_fn(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    # trackers + speed + counts
    tracker = YOLOByteTrackWrapper(model_path=args.yolo_weights, conf=args.conf, device=args.device)
    speed = SpeedEstimator(fps=fps, pixel_per_meter=ppm)
    counter = LineCounter((line_px[0], line_px[1]), (line_px[2], line_px[3]))

    # CLIP
    clip = CLIPZeroShot(model_id=clip_id, device=args.device)
    clip.load()

    labels = [s.strip() for s in (args.labels.split(',') if args.labels else CANDIDATE_LABELS_DEFAULT) if s.strip()]

    processed = 0
    for frame, tracks in tracker.iter_tracks(args.video):
        # speed + counts
        speed_kmh = speed.update(tracks)
        left_to_right, right_to_left = counter.update(tracks)

        draw_counting_line(frame, line_px, color=(0, 255, 0))

        # classify each track crop
        for t in tracks:
            tid = t['id']
            v = speed_kmh.get(tid, 0.0)
            if v < args.min_speed:
                continue
            x1, y1, x2, y2 = map(int, t['bbox'])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W - 1, x2), min(H - 1, y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size <= 0:
                continue

            try:
                best_label, best_prob, _ = clip.classify(crop, labels)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                draw_label(frame, f'ID {tid} | {best_label} ({best_prob:.2f}) | {v:.1f} km/h',
                           x1, max(20, y1 - 10))
            except Exception as e:
                # If CLIP fails (e.g., missing GPU or model not cached), still draw box & speed
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                draw_label(frame, f'ID {tid} | {v:.1f} km/h', x1, max(20, y1 - 10))

        draw_label(frame, f'Count L->R: {left_to_right}', 10, 30)
        draw_label(frame, f'Count R->L: {right_to_left}', 10, 60)

        writer.write(frame)
        processed += 1
        if total_frames and processed % max(1, total_frames // 100) == 0:
            pct = processed / total_frames * 100.0
            print(f'Processing: {processed}/{total_frames} ({pct:.1f}%)', end='\r', flush=True)

    print()
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
