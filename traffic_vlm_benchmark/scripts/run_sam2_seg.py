#!/usr/bin/env python
import sys, argparse, yaml
from pathlib import Path
import cv2, numpy as np, os

def _try_sam2(model_id: str, device: str | None):
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2 = build_sam2(model_id)
        pred = SAM2ImagePredictor(sam2)
        return ("sam2", pred)
    except Exception:
        return (None, None)

def _try_sam_v1(model_id: str, device: str | None):
    try:
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        mname = 'vit_h' if 'huge' in model_id.lower() else 'vit_l'
        sam = sam_model_registry[mname](checkpoint=model_id)
        gen = SamAutomaticMaskGenerator(sam)
        return ("sam1", gen)
    except Exception:
        return (None, None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/sam')
    ap.add_argument('--device', default=None)
    ap.add_argument('--model-id', default=None, help='override config.models.sam2 (or path to SAM v1 ckpt)')
    ap.add_argument('--stride', type=int, default=1)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    model_id = args.model_id or (cfg.get('models', {}) or {}).get('sam2', 'facebook/sam2')

    kind, predictor = _try_sam2(model_id, args.device)
    if kind is None:
        kind, predictor = _try_sam_v1(model_id, args.device)
    if kind is None:
        raise SystemExit(
            "Neither SAM2 nor SAM v1 is available.\n"
            "Install one of:\n"
            "  pip install sam2  (and download a SAM2 checkpoint) \n"
            "  pip install segment-anything  (and provide a SAM v1 checkpoint via --model-id)"
        )

    print(f"[info] Using backend: {kind} | model: {model_id} | device: {args.device or 'cpu'}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cfg.get('fps', cap.get(cv2.CAP_PROP_FPS) or 30)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Fallback: estimate from duration if possible, else unknown
        total_frames = -1
    if total_frames > 0:
        print(f"[info] Total frames: {total_frames}")
    else:
        print("[warn] Could not determine total frame count; progress % will be approximate.")
    os.makedirs(args.out, exist_ok=True)
    out_path = str(Path(args.out) / (Path(args.video).stem + f"_{kind}.mp4"))
    # cv2.VideoWriter_fourcc may not exist in some builds/stubs; compute FOURCC manually as a fallback.
    if hasattr(cv2, 'VideoWriter_fourcc'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = (ord('m') | (ord('p') << 8) | (ord('4') << 16) | (ord('v') << 24))
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    if not writer.isOpened():
        raise SystemExit(f"Failed to open video writer for: {out_path}")
    print(f"[info] Writing to: {out_path} @ {fps} FPS, size {W}x{H}")

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if idx % args.stride != 0:
            writer.write(frame); idx += 1; continue

        if kind == "sam2":
            # SAM2 predictor expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ensure predictor is set
            assert predictor is not None
            # Try several possible SAM2 predictor APIs explicitly with the image argument,
            # falling back to set_image()+generate() if necessary.
            masks = None
            try:
                if hasattr(predictor, 'generate'):
                    # prefer passing image explicitly (some SAM2 builds expect this)
                    try:
                        masks = predictor.generate(rgb)  # type: ignore[attr-defined]
                    except TypeError:
                        # some versions expect set_image() first and then generate() with no args
                        predictor.set_image(rgb)  # type: ignore[attr-defined]
                        masks = predictor.generate()  # type: ignore[attr-defined]
                elif hasattr(predictor, 'generate_masks'):
                    masks = predictor.generate_masks(rgb)  # type: ignore[attr-defined]
                elif hasattr(predictor, 'predict'):
                    masks = predictor.predict(rgb)  # type: ignore[attr-defined]
                else:
                    # last resort: try set_image + generate
                    predictor.set_image(rgb)  # type: ignore[attr-defined]
                    masks = predictor.generate()  # type: ignore[attr-defined]
            except Exception:
                masks = None

            overlay = frame.copy()
            occ = 0.0
            if masks:
                last_msk = None
                for m in masks:
                    # support both dict-based masks and objects with a 'segmentation' attribute
                    if isinstance(m, dict):
                        seg = m.get('segmentation')
                    else:
                        seg = getattr(m, 'segmentation', None)
                    if seg is None:
                        continue
                    seg_arr = seg.astype(np.uint8) * 255
                    last_msk = seg_arr
                    color = (0,255,0)
                    overlay[seg_arr>0] = (overlay[seg_arr>0]*0.5 + np.array(color)*0.5).astype(np.uint8)
                if last_msk is not None:
                    occ = float(np.count_nonzero(last_msk)) / float(last_msk.size) * 100.0
            out = overlay
        else:
            # SAM v1 auto generator
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            masks = predictor.generate(rgb)
            overlay = frame.copy()
            occ = 0.0
            for m in masks[:30]:
                msk = m['segmentation'].astype(np.uint8)*255
                overlay[msk>0] = (overlay[msk>0]*0.5 + np.array([0,255,0])*0.5).astype(np.uint8)
                occ += np.count_nonzero(msk)
            occ = (occ / (frame.shape[0]*frame.shape[1]))*100.0
            out = overlay

        # Overlay occupancy
        cv2.putText(out, f"Occupancy ~ {occ:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),3,cv2.LINE_AA)
        cv2.putText(out, f"Occupancy ~ {occ:.1f}%", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),1,cv2.LINE_AA)
        # Overlay progress percent if total known
        if total_frames > 0:
            pct = (idx / total_frames) * 100.0
            cv2.putText(out, f"Progress {pct:.1f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(out, f"Progress {pct:.1f}%", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),1,cv2.LINE_AA)
        writer.write(out)
        if idx % 60 == 0:
            if total_frames > 0:
                print(f"[info] Processed {idx}/{total_frames} frames ({(idx/total_frames)*100:.1f}%)")
            else:
                print(f"[info] Processed {idx} frames...")
        idx += 1

    writer.release(); cap.release()
    if total_frames > 0:
        print(f"[info] Completed {idx}/{total_frames} frames (100%).")
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
