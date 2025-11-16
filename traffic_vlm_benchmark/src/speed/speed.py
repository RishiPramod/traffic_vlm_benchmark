import numpy as np
from collections import defaultdict

class SpeedEstimator:
    def __init__(self, fps, pixel_per_meter):
        self.fps = fps
        self.ppm = pixel_per_meter
        self.prev_centers = {}  # id -> (x,y)
        self.speeds = defaultdict(lambda: 0.0)

    def update(self, tracks_frame):
        # tracks_frame: list of dicts: {'id':int, 'bbox':[x1,y1,x2,y2]}
        for t in tracks_frame:
            tid = t['id']
            x1,y1,x2,y2 = t['bbox']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            if tid in self.prev_centers:
                px, py = self.prev_centers[tid]
                dist_px = ((cx - px)**2 + (cy - py)**2)**0.5
                dist_m = dist_px / max(self.ppm, 1e-6)
                # speed in m/s -> km/h
                v_ms = dist_m * self.fps
                v_kmh = v_ms * 3.6
                self.speeds[tid] = 0.8*self.speeds[tid] + 0.2*v_kmh  # EMA smoothing
            self.prev_centers[tid] = (cx, cy)
        return dict(self.speeds)

