#!/usr/bin/env python
import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import yaml

YOLO_ALIASES = {'yolo11l.pt'}

def _is_yolo_weight(v: str) -> bool:
    v = (v or '').strip().lower()
    return v.endswith(('.pt', '.pth')) or v in YOLO_ALIASES

def _run(cmd):
    print('Running:', ' '.join(cmd))
    return subprocess.run(cmd, check=False).returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/demo')
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--min-speed', type=float, default=1.0)
    ap.add_argument('--device', default=None)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--model-keys', default=None,
                    help='comma-separated keys from config.models to run '
                         '(e.g., "yolo11l,clip"). If omitted, auto-selects YOLO entries and clip if present.')
    args = ap.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, 'r'))
    models: Dict[str, Any] = cfg.get('models', {}) or {}
    if not models:
        raise SystemExit('No models found in config["models"].')

    # build selection
    selected = {}
    if args.model_keys:
        req = [k.strip() for k in args.model_keys.split(',') if k.strip()]
        for k in req:
            if k not in models:
                print(f'Warning: {k} not in config.models; skipping')
                continue
            selected[k] = models[k]
    else:
        # auto: include YOLO entries and CLIP (if present)
        for k, v in models.items():
            if _is_yolo_weight(str(v)) or k.lower() == 'clip':
                selected[k] = v

    if not selected:
        raise SystemExit('Nothing to run. Add YOLO .pt entries or specify --model-keys.')

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    video_stem = Path(args.video).stem
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_conf = str(round(float(args.conf), 2)).replace('.', 'p')
    safe_min = str(round(float(args.min_speed), 2)).replace('.', 'p')

    tasks = []
    for key, val in selected.items():
        key_l = key.lower()
        if key_l == 'clip':
            # needs a YOLO backbone; prefer yolo11l if present
            yolo_w = str(models.get('yolo11l', 'yolo11l.pt'))
            out_name = f"{video_stem}_yolo+clip_conf{safe_conf}_min{safe_min}_{ts}.mp4"
            cmd = [
                shutil.which('python') or 'python',
                'scripts/run_clip_with_yolo.py',
                '--video', args.video,
                '--config', args.config,
                '--out', str(out_dir),
                '--conf', str(args.conf),
                '--min-speed', str(args.min_speed),
                '--yolo-weights', yolo_w,
                '--clip-model', str(val),
                '--out-name', out_name
            ]
            if args.device:
                cmd += ['--device', args.device]
            tasks.append((key, cmd))
        elif _is_yolo_weight(str(val)):
            model_name = Path(str(val)).stem
            out_name = f"{video_stem}_{model_name}_conf{safe_conf}_min{safe_min}_{ts}.mp4"
            cmd = [
                shutil.which('python') or 'python',
                'scripts/run_yolo11_pipeline.py',
                '--video', args.video,
                '--config', args.config,
                '--out', str(out_dir),
                '--conf', str(args.conf),
                '--min-speed', str(args.min_speed),
                '--yolo-weights', str(val),
                '--out-name', out_name
            ]
            if args.device:
                cmd += ['--device', args.device]
            tasks.append((key, cmd))
        else:
            print(f"Skipping '{key}': needs its own runner (e.g., Grounding DINO, SAM2, InternVL2, VideoLLaMA-2).")

    # run
    failures = []
    if args.parallel > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {ex.submit(_run, cmd): key for key, cmd in tasks}
            for fut in as_completed(futs):
                key = futs[fut]
                try:
                    rc = fut.result()
                    if rc != 0:
                        failures.append((key, rc))
                except Exception as e:
                    failures.append((key, str(e)))
    else:
        for key, cmd in tasks:
            rc = _run(cmd)
            if rc != 0:
                failures.append((key, rc))

    if failures:
        print('Some runs failed:')
        for f in failures:
            print(' ', f)
        raise SystemExit(1)

    print('All runs completed successfully.')

if __name__ == '__main__':
    main()
