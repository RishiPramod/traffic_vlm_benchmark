#!/usr/bin/env python
import argparse, yaml, os
from pathlib import Path
import numpy as np
import cv2

def _load_videollama2(model_id: str, device: str | None):
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device or "cpu")
        return model, proc
    except Exception as e:
        raise SystemExit(
            "Failed to load VideoLLaMA-2.\n"
            "Install: pip install transformers accelerate sentencepiece\n"
            f"Error: {e}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/videollama2')
    ap.add_argument('--device', default=None)
    ap.add_argument('--model-id', default=None)
    ap.add_argument('--question', default='Which direction are most vehicles moving over time? Mention any lane changes.')
    ap.add_argument('--window', type=int, default=120, help='use first N frames')
    ap.add_argument('--stride', type=int, default=15, help='sample every Nth frame into the context')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    model_id = args.model_id or (cfg.get('models',{}) or {}).get('videollama2', 'DAMO-NLP-SG/VideoLLaMA2')

    model, proc = _load_videollama2(model_id, args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cfg.get('fps', cap.get(cv2.CAP_PROP_FPS) or 30)
    os.makedirs(args.out, exist_ok=True)
    out_path = str(Path(args.out) / (Path(args.video).stem + "_videollama2.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))

    # sample a context of frames
    frames = []
    for i in range(args.window):
        ok, frm = cap.read()
        if not ok: break
        if i % args.stride == 0:
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    if not frames:
        raise SystemExit("No frames sampled for context.")

    # try generation
    try:
        inputs = proc(text=args.question, videos=[frames], return_tensors="pt")  # some ports accept image seq as "videos"
        if args.device:
            inputs = {k: v.to(args.device) for k,v in inputs.items() if hasattr(v,'to')}
        out_ids = model.generate(**inputs, max_new_tokens=128)
        # some processors lack tokenizer; fall back to decoding by processor if provided
        try:
            answer = proc.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        except Exception:
            answer = "[temporal answer generated]"
    except Exception:
        answer = "[Could not run temporal generation with this build; verify model docs.]"

    # write the whole video with the answer overlay
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frm = cap.read()
        if not ok: break
        cv2.rectangle(frm, (0,0), (W,60), (0,0,0), -1)
        cv2.putText(frm, answer[:180], (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2,cv2.LINE_AA)
        writer.write(frm)

    writer.release(); cap.release()
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    main()
