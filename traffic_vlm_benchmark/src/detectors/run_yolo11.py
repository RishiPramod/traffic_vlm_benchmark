#!/usr/bin/env python
"""Compatibility wrapper so users can run the detector entrypoint from
`src/detectors/run_yolo11.py` (some older commands expect this path).
This simply delegates to the existing `scripts/run_yolo11_pipeline.py` main().
"""
import sys
from pathlib import Path

# Ensure repo root is on sys.path so we can import scripts.run_yolo11_pipeline
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from scripts.run_yolo11_pipeline import main
except Exception as e:
    raise

if __name__ == '__main__':
    main()
