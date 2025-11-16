# TODO: plug actual GroundingDINO pipeline using groundingdino-py or transformers
# This is a placeholder that defines the interface expected by the evaluation harness.

class GroundingDINOTextDetector:
    def __init__(self, model_id='groundingdino-tiny', device=None, box_threshold=0.25, text_threshold=0.25):
        self.model_id = model_id
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

    def detect_with_text(self, frame_bgr, prompt: str):
        """Return list of dicts with bboxes filtered by text prompt.
        Each dict: {'bbox':[x1,y1,x2,y2], 'label': str, 'score': float}
        """
        # TODO: implement actual inference
        return []
