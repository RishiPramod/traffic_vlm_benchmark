#!/usr/bin/env python
r"""Top-level convenience wrapper to run the batch model pipeline from the repo root.
Delegates to `traffic_vlm_benchmark/scripts/run_all_models.py`.
Usage from repo root (PowerShell):
    python .\scripts\run_all_models.py --video Converted_videos_mp4/005cdef0-180a776c.mp4 --config traffic_vlm_benchmark/configs/config.yaml --out runs/demo
Forward any additional arguments (e.g. --model-keys, --parallel) and they will
be handled by the underlying script.
"""
import sys
from pathlib import Path

# Resolve package root (the directory containing the internal scripts folder)
REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = REPO_ROOT / 'traffic_vlm_benchmark'
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from traffic_vlm_benchmark.scripts.run_all_models import main  # type: ignore

if __name__ == '__main__':
    main()
