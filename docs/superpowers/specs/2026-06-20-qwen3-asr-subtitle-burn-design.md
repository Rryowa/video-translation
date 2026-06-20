# Qwen3-ASR Subtitle Generation and Burning

## Architecture
- **Language**: Python
- **Environment**: Windows PowerShell
- **Libraries**: `torch`, `qwen-asr` (or custom transformers pipeline), `ffmpeg-python`

## Components
1. **Audio Extractor**: Uses FFmpeg to extract 16kHz mono audio from input video.
2. **ASR Transcriber**: Uses `Qwen3ASRModel` to transcribe audio.
3. **Forced Aligner**: Uses `Qwen3-ForcedAligner` to extract precise word-level timestamps.
4. **SRT Generator**: Groups words into segments (by pause/punctuation) and formats as standard `.srt`.
5. **Subtitle Burner**: Uses FFmpeg (`-vf subtitles=...`) to burn `.srt` into the video. *Note: Windows path escaping required for subtitle filter.*

## Data Flow
Input Video → Extracted Audio (.wav) → Qwen3 ASR → Timestamps → SRT File → FFmpeg Burn → Output Video
