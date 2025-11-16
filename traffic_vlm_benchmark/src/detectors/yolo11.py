from ultralytics import YOLO
import numpy as np

class YOLO11Detector:
    def __init__(self, model_path='yolo11l.pt', device=None, conf=0.3, classes=None):
        self.model = YOLO(model_path)
        self.device = device
        self.conf = conf
        self.classes = classes  # list of class ids to keep, or None for all

    def detect(self, frame):
        # returns list of dicts: {bbox:[x1,y1,x2,y2], cls:int, conf:float}
        results = self.model(frame, verbose=False, device=self.device, conf=self.conf, classes=self.classes)
        dets = []
        if len(results):
            r = results[0]
            boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0,4))
            clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros((0,))
            confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros((0,))
            for b, c, s in zip(boxes, clss, confs):
                dets.append({'bbox': b.tolist(), 'cls': int(c), 'conf': float(s)})
        return dets

    def track(self, source, device=None, conf=None, classes=None, project='runs', name='yolo11_track'):
        # Convenience: leverage Ultralytics built-in tracker (ByteTrack) for a quick demo
        # Output video is saved under runs/ by Ultralytics.
        self.model.track(source=source, device=device or self.device, conf=conf or self.conf,
                         classes=classes or self.classes, tracker='bytetrack.yaml', project=project, name=name, save=True, verbose=False)
