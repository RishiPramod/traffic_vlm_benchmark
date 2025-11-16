#!/usr/bin/env python
# traffic_vlm_benchmark/scripts/run_grounding_dino_query.py

import sys
import argparse
import yaml
from pathlib import Path
import cv2
import numpy as np
import os
import tempfile
from typing import Optional, Tuple


# Add repo root so src.* imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import draw_label  # already in your repo


def _load_gdino(model_id: str, device: Optional[str]) -> Tuple[str, tuple]:
    """
    Try to load GroundingDINO via groundingdino-py first (preferred).
    If unavailable, try a very light HF placeholder that just warns.
    Returns: (kind, handle_tuple)
      - kind == "gdino_py": handle = (load_model, predict)
      - kind == "hf":      handle = (model, processor_or_none)
    """
    # groundingdino-py
    try:
        import importlib
        inf = importlib.import_module("groundingdino.util.inference")
        load_model = getattr(inf, "load_model")
        predict = getattr(inf, "predict")
        return ("gdino_py", (load_model, predict))
    except Exception:
        pass

    # Transformers fallback (placeholder; most ports are not drop-in here)
    try:
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(model_id).to(device or "cpu")
        proc = AutoProcessor.from_pretrained(model_id)
        return ("hf", (model, proc))
    except Exception as e:
        raise SystemExit(
            "GroundingDINO not available.\n"
            "Install the official binding:\n"
            "  pip install groundingdino-py\n"
            "Or use a supported HF port and adapt this script.\n"
            f"Underlying error: {e}"
        )


def _fourcc_safe() -> int:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if fourcc_fn is None:
        fourcc_fn = getattr(getattr(cv2, "cv2", cv2), "VideoWriter_fourcc")
    return fourcc_fn(*"mp4v")


def _ensure_paths(cfg_path: Optional[str], ckpt_path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Validate/normalize config and checkpoint paths for groundingdino-py:
      - cfg_path must be a .py file that exists; if YAML, convert to temp .py.
      - ckpt_path should be a .pth file; if it's .safetensors, convert to a temp .pth.
    Returns normalized (cfg_path_str, ckpt_path_str_or_None).
    """
    if cfg_path is None:
        raise SystemExit(
            "GroundingDINO requires a Python config file.\n"
            "Pass --gdino-config PATH (e.g., GroundingDINO_SwinT_OGC.py or .yaml)."
        )

    cfg_p = Path(cfg_path).expanduser().resolve()
    if not cfg_p.exists():
        raise SystemExit(f"GroundingDINO config not found: {cfg_p}")

    if cfg_p.suffix.lower() == ".yaml" or cfg_p.suffix.lower() == ".yml":
        # Convert YAML to temp .py
        try:
            import yaml
            with open(cfg_p, 'r') as f:
                config_dict = yaml.safe_load(f)
            # Create a temp .py file with the config as Python variables
            import tempfile
            temp_py = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
            temp_py.write("# Auto-generated from YAML config\n")
            for k, v in config_dict.items():
                temp_py.write(f"{k} = {repr(v)}\n")
            temp_py.close()
            # Store temp path for cleanup
            globals()["_GDINO_TEMP_PY"] = temp_py.name
            cfg_p = Path(temp_py.name)
        except Exception as e:
            raise SystemExit(f"Failed to convert YAML config '{cfg_p}': {e}")

    if cfg_p.suffix.lower() != ".py":
        raise SystemExit(
            f"GroundingDINO config must be a .py file: {cfg_p}\n"
            "Use the official .py config, e.g., GroundingDINO_SwinT_OGC.py"
        )

    ckpt_str: Optional[str] = None
    if ckpt_path:
        ckpt_p = Path(ckpt_path).expanduser().resolve()
        if not ckpt_p.exists():
            if ckpt_p.is_dir():
                pths = list(ckpt_p.glob("*.pth"))
                if not pths:
                    raise SystemExit(f"No .pth checkpoint found in directory: {ckpt_p}")
                ckpt_p = pths[0]
            else:
                raise SystemExit(
                    f"GroundingDINO checkpoint not found: {ckpt_p}\n"
                    "Provide a .pth file, e.g., groundingdino_swint_ogc.pth"
                )

        if ckpt_p.suffix.lower() == ".safetensors":
            # Convert to .pth for groundingdino-py
            try:
                from safetensors.torch import load_file as st_load
                import torch
                sd = st_load(str(ckpt_p))
                tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
                tmpf.close()
                torch.save(sd, tmpf.name)
                # Store temp path on module global so we can cleanup after run
                globals()["_GDINO_TEMP_PTH"] = tmpf.name
                ckpt_str = tmpf.name
            except Exception as e:
                raise SystemExit(
                    f"Failed to convert safetensors checkpoint '{ckpt_p}': {e}\n"
                    "Install 'safetensors' and 'torch', or use a native .pth checkpoint."
                )
        else:
            ckpt_str = str(ckpt_p)

    return (str(cfg_p), ckpt_str)


def _parse_prompt(text: str) -> str:
    """
    Normalize prompt (GroundingDINO typically expects a comma-separated list).
    """
    if not text:
        return "car, bus, truck, motorcycle"
    # allow arbitrary separators; normalize to comma + space
    parts = [p.strip() for p in text.replace("|", ",").split(",") if p.strip()]
    return ", ".join(parts) if parts else "car, bus, truck, motorcycle"


def draw_boxes(frame, boxes, labels, color=(0, 255, 0)):
    for (x1, y1, x2, y2), lab in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        draw_label(frame, str(lab), x1, max(20, y1 - 10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True, help='input video path')
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/gdino')
    ap.add_argument('--device', default=None)
    ap.add_argument('--text', default='car, bus, truck, motorcycle')
    ap.add_argument('--model-id', default=None,
                    help='GroundingDINO checkpoint path (.pth) or HF repo id (HF path is placeholder).')
    ap.add_argument('--gdino-config', default=None,
                    help='Path to GroundingDINO .py config (e.g., GroundingDINO_SwinT_OGC.py) for groundingdino-py.')
    ap.add_argument('--box-thr', type=float, default=0.25)
    ap.add_argument('--text-thr', type=float, default=0.25)
    ap.add_argument('--stride', type=int, default=1, help='process every Nth frame')
    args = ap.parse_args()

    # Load general config for fps default
    cfg_all = yaml.safe_load(open(args.config, 'r'))
    fps_cfg = cfg_all.get('fps', None)

    # Resolve model id from config if not provided
    if args.model_id is None:
        models_cfg = cfg_all.get('models', {}) or {}
        args.model_id = models_cfg.get('grounding_dino_tiny', 'groundingdino-tiny')

    kind, handle = _load_gdino(args.model_id, args.device)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps_cfg or (cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Prepare writer
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = str(out_dir / (Path(args.video).stem + "_groundingdino.mp4"))
    writer = cv2.VideoWriter(out_path, _fourcc_safe(), float(fps), (W, H))

    text_prompt = _parse_prompt(args.text)

    # Warm up/load model for gdino_py
    temp_pth_to_cleanup = None
    if kind == "gdino_py":
        load_model, predict = handle

        # Validate and normalize config & checkpoint paths
        cfg_path_str, ckpt_path_str = _ensure_paths(args.gdino_config, args.model_id)
        if "_GDINO_TEMP_PTH" in globals():
            temp_pth_to_cleanup = globals()["_GDINO_TEMP_PTH"]

        # Cache model on function attribute so reload doesn't happen each frame
        main._gdino_model = load_model(
            model_config_path=cfg_path_str,
            model_checkpoint_path=ckpt_path_str,
            device=args.device
        )

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_idx % args.stride != 0:
            writer.write(frame)
            frame_idx += 1
            continue

        if kind == "gdino_py":
            # Convert BGR->RGB for inference
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                boxes, logits, phrases = predict(
                    model=main._gdino_model,
                    image=image_rgb,
                    caption=text_prompt,
                    box_threshold=args.box_thr,
                    text_threshold=args.text_thr
                )
                print(f"Frame {frame_idx}: {len(boxes)} detections")
                draw_boxes(frame, boxes, phrases)
            except Exception as e:
                print(f"Frame {frame_idx}: Prediction error: {e}")
                draw_label(frame, f'GroundingDINO error: {e}', 10, 30)

        else:
            # HF placeholder: we don't execute detection here because HF ports differ;
            # give a clear on-frame message instead of crashing.
            draw_label(frame, 'HF GroundingDINO port not implemented in this runner.', 10, 30)
            draw_label(frame, f'Prompt: {text_prompt}', 10, 60)

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()

    # Cleanup temp .pth if we created one
    if temp_pth_to_cleanup and Path(temp_pth_to_cleanup).exists():
        try:
            os.remove(temp_pth_to_cleanup)
        except Exception:
            pass

    # Cleanup temp .py if we created one from YAML
    if "_GDINO_TEMP_PY" in globals():
        temp_py_path = globals()["_GDINO_TEMP_PY"]
        if Path(temp_py_path).exists():
            try:
                os.remove(temp_py_path)
            except Exception:
                pass

    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()
