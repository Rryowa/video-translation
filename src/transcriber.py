import torch
import logging
from qwen_asr import Qwen3ASRModel

logger = logging.getLogger("transcriber")
model = None

def get_model():
    global model
    if model is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        
        logger.debug(f"Loading Qwen3-ASR model on {device} with dtype {dtype}...")
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
        logger.debug("Models successfully loaded.")
    return model

def transcribe_audio(audio_path: str):
    logger.debug(f"Starting ASR transcription for {audio_path}")
    m = get_model()
    # return_time_stamps=True gets ForcedAlignResult in time_stamps
    results = m.transcribe(audio_path, return_time_stamps=True)
    logger.debug("ASR transcription finished.")
    return results[0]

def free_model():
    global model
    if model is not None:
        logger.debug("Deleting transcriber model and freeing CUDA cache...")
        del model
        model = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("Memory release completed.")
