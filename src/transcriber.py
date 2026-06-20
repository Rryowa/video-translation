import torch
from qwen_asr import Qwen3ASRModel

model = None

def get_model():
    global model
    if model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        # Load the model with forced aligner
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            dtype=dtype, 
            device_map=device,
            forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
            forced_aligner_kwargs=dict(
                dtype=dtype,
                device_map=device
            )
        )
    return model

def transcribe_audio(audio_path: str):
    m = get_model()
    # return_time_stamps=True gets ForcedAlignResult in time_stamps
    results = m.transcribe(audio_path, return_time_stamps=True)
    return results[0]
