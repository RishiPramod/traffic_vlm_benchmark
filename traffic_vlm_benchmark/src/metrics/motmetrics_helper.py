import motmetrics as mm

class MOTAccumulator:
    def __init__(self):
        self.acc = mm.MOTAccumulator(auto_id=True)

    def update(self, gt_ids, gt_boxes, pred_ids, pred_boxes):
        # cost matrix using IOU distance
        import numpy as np
        def bbox_center(b):
            x1,y1,x2,y2 = b
            return ((x1+x2)/2.0, (y1+y2)/2.0)
        # Use 1-IOU as distance
        def iou(a,b):
            x1 = max(a[0], b[0])
            y1 = max(a[1], b[1])
            x2 = min(a[2], b[2])
            y2 = min(a[3], b[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
            area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
            union = area_a + area_b - inter + 1e-6
            return inter/union
        import numpy as np
        if len(gt_boxes)==0 or len(pred_boxes)==0:
            dists = np.array([[]])
        else:
            dists = np.zeros((len(gt_boxes), len(pred_boxes)))
            for i, g in enumerate(gt_boxes):
                for j, p in enumerate(pred_boxes):
                    dists[i, j] = 1.0 - iou(g, p)
        self.acc.update(gt_ids, pred_ids, dists)

    def summary(self):
        mh = mm.metrics.create()
        return mh.compute(self.acc, metrics=['mota','idf1','num_switches','precision','recall'], name='acc')
