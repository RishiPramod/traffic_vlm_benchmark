import cv2
import numpy as np

def denorm_line(line_norm, w, h):
    x1 = int(line_norm[0] * w)
    y1 = int(line_norm[1] * h)
    x2 = int(line_norm[2] * w)
    y2 = int(line_norm[3] * h)
    return (x1, y1, x2, y2)

def draw_counting_line(frame, line_px, color=(0,255,0)):
    x1,y1,x2,y2 = line_px
    cv2.line(frame, (x1,y1), (x2,y2), color, 2)
    cv2.circle(frame, (x1,y1), 4, color, -1)
    cv2.circle(frame, (x2,y2), 4, color, -1)

def draw_label(frame, text, x, y):
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

def iou(a, b):
    # boxes: [x1,y1,x2,y2]
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-6
    return inter / union
