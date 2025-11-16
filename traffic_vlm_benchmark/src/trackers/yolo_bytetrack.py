from pathlib import Path
from ultralytics import YOLO

class YOLOByteTrackWrapper:
    """Wrap Ultralytics .track() to yield (frame_bgr, [{'id': int, 'bbox': [x1,y1,x2,y2]}, ...])."""

    def __init__(self, model_path='yolo11l.pt', conf=0.3, device=None, classes=None):
        model_path = str(model_path)

        is_file = Path(model_path).exists()
        is_weights = model_path.lower().endswith(('.pt', '.pth'))
        is_alias = model_path.lower() in {'yolo11l.pt'}

        if not (is_file and is_weights) and not (is_weights and not is_file) and not is_alias:
            raise ValueError(
                f"YOLOByteTrackWrapper expected a YOLO weights file (.pt/.pth) or alias, got: '{model_path}'.\n"
                f"Do NOT pass HF repo IDs here (e.g., 'openai/clip-vit-base-patch32').\n"
                f"Use --yolo-weights PATH or set config.models.yolo11l to a .pt file/alias."
            )

        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.classes = classes

    def iter_tracks(self, video_path):
        results = self.model.track(
            source=video_path,
            device=self.device,
            conf=self.conf,
            classes=self.classes,
            tracker='bytetrack.yaml',
            stream=True,
            verbose=False
        )
        for r in results:
            frame = r.orig_img  # BGR
            tracks = []
            if r.boxes is not None and r.boxes.id is not None:
                ids = r.boxes.id.int().cpu().tolist()
                boxes = r.boxes.xyxy.cpu().tolist()
                for i, b in zip(ids, boxes):
                    tracks.append({'id': int(i), 'bbox': b})
            yield frame, tracks
