# TODO: integrate InternVL2 via transformers + generate answers/explanations
class InternVL2VLM:
    def __init__(self, model_id='OpenGVLab/InternVL2-8B', device=None):
        self.model_id = model_id
        self.device = device

    def answer(self, frames, question: str) -> str:
        # frames: list of BGR images
        return "[stub]"
