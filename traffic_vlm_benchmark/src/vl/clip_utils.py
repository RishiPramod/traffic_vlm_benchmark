# Simple CLIP wrapper for zero-shot labeling of crops
# Requires: transformers, torch

class CLIPZeroShot:
    def __init__(self, model_id='openai/clip-vit-base-patch32', device=None):
        self.model_id = model_id
        self.device = device
        self._model = None

    def load(self):
        from transformers import CLIPProcessor, CLIPModel
        self._model = CLIPModel.from_pretrained(self.model_id)
        self._processor = CLIPProcessor.from_pretrained(self.model_id)
        if self.device:
            self._model.to(self.device)

    def classify(self, image_bgr, candidate_labels):
        if self._model is None:
            self.load()
        import cv2
        import torch
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(text=candidate_labels, images=image_rgb, return_tensors='pt', padding=True)
        if self.device:
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits_per_image[0]
            probs = logits.softmax(dim=0).cpu().tolist()
        best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
        return candidate_labels[best_idx], probs[best_idx], probs
