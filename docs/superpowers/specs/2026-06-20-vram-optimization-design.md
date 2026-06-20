# VRAM Optimization Design

Release GPU memory held by the ASR and Forced Aligner models before starting translation to make VRAM available for the local LLM.

## Background
The ASR model (`Qwen3-ASR-1.7B`) and Forced Aligner (`Qwen3-ForcedAligner-0.6B`) together consume around 4.6 GB of VRAM. If translation is enabled, the system connects to a local LLM (e.g. `llama.cpp`) which requires loading a large model (e.g., Qwen3.5-9B Q8, consuming ~11 GB of VRAM). Holding ASR VRAM in memory concurrently may cause Out-Of-Memory (OOM) errors on the GPU.

## Proposed Changes

### 1. `src/transcriber.py`
- Add a new helper function `free_model()`.
- Set the global `model` variable to `None` and delete references.
- Trigger Python garbage collection (`gc.collect()`).
- Empty the PyTorch CUDA cache (`torch.cuda.empty_cache()`) if CUDA is available.

### 2. `main.py`
- Import `free_model` from `src.transcriber`.
- Call `free_model()` right after transcription grouping is completed, freeing VRAM before translation API calls are initiated.
