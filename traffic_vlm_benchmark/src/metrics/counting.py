import shapely.geometry as geom

class LineCounter:
    def __init__(self, p1, p2):
        self.line = geom.LineString([p1, p2])
        self.side = {}  # id -> side
        self.count_lr = 0
        self.count_rl = 0

    def update(self, tracks_frame):
        # tracks_frame: list of dicts: {'id':int, 'bbox':[x1,y1,x2,y2]}
        for t in tracks_frame:
            tid = t['id']
            x1,y1,x2,y2 = t['bbox']
            cx, cy = (x1+x2)/2, (y1+y2)/2
            pt = geom.Point(cx, cy)
            # Determine side via signed distance using line endpoints
            xA,yA = self.line.coords[0]
            xB,yB = self.line.coords[1]
            side = (xB - xA)*(cy - yA) - (yB - yA)*(cx - xA)
            side = 1 if side > 0 else -1
            if tid not in self.side:
                self.side[tid] = side
            else:
                if self.side[tid] != side:
                    # crossed
                    if side > 0:
                        self.count_rl += 1
                    else:
                        self.count_lr += 1
                    self.side[tid] = side
        return self.count_lr, self.count_rl
