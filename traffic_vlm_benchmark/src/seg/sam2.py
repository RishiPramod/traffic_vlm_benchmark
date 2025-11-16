# TODO: integrate SAM2 for segmentation masks and occupancy maps
class SAM2Segmenter:
    def __init__(self, model_id='facebook/sam2', device=None):
        self.model_id = model_id
        self.device = device

    def segment(self, frame_bgr):
        # return list of masks; placeholder
        return []
