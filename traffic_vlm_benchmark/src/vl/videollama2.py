# TODO: integrate VideoLLaMA-2 for temporal reasoning on sequences
class VideoLLaMA2:
    def __init__(self, model_id='DAMO-NLP-SG/VideoLLaMA2', device=None):
        self.model_id = model_id
        self.device = device

    def describe_event(self, frames, prompt: str) -> str:
        return "[stub]"
