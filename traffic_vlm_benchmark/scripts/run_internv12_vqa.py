#!/usr/bin/env python
import argparse, yaml, os
from pathlib import Path
import numpy as np
import cv2

def _load_internvl2(model_id: str, device: str | None):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

        # Tokenizer/processor first
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # Memory-aware loading
        use_cuda = bool(device and str(device).startswith("cuda"))
        dtype = torch.float16 if use_cuda else None
        device_map = "auto" if use_cuda else None

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )

        if device_map is None:
            # Explicit placement for CPU or single device
            dev = torch.device(device or "cpu")
            model = model.to(dev)

        model.eval()
        return model, tok, proc
    except Exception as e:
        raise SystemExit(
            "Failed to load InternVL2.\n"
            "Install: pip install transformers accelerate sentencepiece\n"
            f"Error: {e}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True)
    ap.add_argument('--config', default='configs/config.yaml')
    ap.add_argument('--out', default='runs/internvl2')
    ap.add_argument('--device', default=None)
    ap.add_argument('--model-id', default=None)
    ap.add_argument('--question', default='Describe the traffic scene and any violations.')
    ap.add_argument('--sample-every', type=int, default=30, help='sample every N frames')
    ap.add_argument('--max-frames', type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config,'r'))
    model_id = args.model_id or (cfg.get('models',{}) or {}).get('internvl2', 'OpenGVLab/InternVL2-8B')

    model, tok, proc = _load_internvl2(model_id, args.device)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cfg.get('fps', cap.get(cv2.CAP_PROP_FPS) or 30))
    os.makedirs(args.out, exist_ok=True)

    # Robust writer creation: try MP4 first, then fallback to AVI if codec unsupported
    out_stem = Path(args.video).stem + "_internvl2"
    out_path = str(Path(args.out) / f"{out_stem}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        # Fallback to AVI with XVID
        fallback_path = str(Path(args.out) / f"{out_stem}.avi")
        fourcc2 = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(fallback_path, fourcc2, fps, (W, H))
        if writer.isOpened():
            print(f"[info] MP4 writer unavailable; fell back to AVI at: {fallback_path}")
            out_path = fallback_path
        else:
            raise SystemExit("Failed to open any video writer (mp4v/XVID). Check codecs/ffmpeg support in OpenCV.")

    # collect a few frames
    frames = []
    idx = 0
    while len(frames) < args.max_frames:
        ok, frm = cap.read()
        if not ok: break
        if idx % args.sample_every == 0:
            frames.append(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
        idx += 1

    if not frames:
        raise SystemExit("No frames sampled.")

    # basic prompt (many VLMs need a chat template + provide text via processor)
    try:
        # 1) Try the processor-first path (expects `text` + `images`)
        prompt = args.question
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": "<image>\n" + args.question}]
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = proc(text=[prompt], images=[frames[0]], return_tensors="pt")
        if args.device:
            dev = args.device
            inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, 'to')}

        out_ids = model.generate(**inputs, max_new_tokens=128)
        answer = tok.decode(out_ids[0], skip_special_tokens=True)
    except Exception as e1:
        print(f"[InternVL2 generation error - processor path] {e1}")
        # 2) Fallback: tokenize text and process image separately, then merge
        try:
            txt = tok([prompt], return_tensors="pt")
            img = proc(images=[frames[0]], return_tensors="pt")
            if args.device:
                dev = args.device
                txt = {k: v.to(dev) for k, v in txt.items() if hasattr(v, 'to')}
                img = {k: v.to(dev) for k, v in img.items() if hasattr(v, 'to')}
            gen_inputs = {**txt, **img}
            out_ids = model.generate(**gen_inputs, max_new_tokens=128)
            answer = tok.decode(out_ids[0], skip_special_tokens=True)
        except Exception as e2:
            print(f"[InternVL2 generation error - fallback path] {e2}")
            answer = "[Could not run full VQA generation; see console for error.]"

    # Render answer on all frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok, frm = cap.read()
        if not ok: break
        cv2.rectangle(frm, (0,0), (W,60), (0,0,0), -1)
        cv2.putText(frm, answer[:180], (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,255,255),2,cv2.LINE_AA)
        writer.write(frm)

    writer.release(); cap.release()

    # Verify file was created and has non-zero size
    try:
        p = Path(out_path)
        if p.exists() and p.stat().st_size > 0:
            print(f"Saved: {p.resolve()}")
        else:
            print(f"[warn] Expected output not found or empty: {p}")
            print("If you're on Windows and MP4 fails, the script should auto-fallback to AVI.\n"
                  "You may also install a FFMPEG-enabled OpenCV build or try opencv-python (non-headless).")
    except Exception as _:
        print(f"Saved (path): {out_path}")

if __name__ == '__main__':
    main()
