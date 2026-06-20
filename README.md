# Video Translation and Subtitling Tool

Automatically extract audio, transcribe with word-level alignment, translate to Russian, and burn subtitles back into a video.

## Features
- **Audio Extraction**: Extracts PCM 16kHz mono audio from video files.
- **ASR & Alignment**: Transcribes audio and forces alignment using Qwen3-ASR (1.7B) and Qwen3-ForcedAligner (0.6B) models.
- **Auto Segmenting**: Intelligently groups aligned tokens into readable subtitle segments based on duration gaps and character limits.
- **Local Translation**: Optional translation via Ollama's native API (`/api/chat`). Automatically forces the model to skip "thinking" steps to ensure strict SRT formatting.
- **Subtitle Burning**: GPU-accelerated rendering (NVENC) to burn the final subtitles back into the output MP4 video.
- **Batch Processing**: Process single videos or entire folders automatically. Output is saved to a `processed/` subfolder, and already processed videos are safely skipped to save time.

## Requirements
- Python 3.10+
- FFmpeg installed and available in system PATH
- CUDA-compatible GPU (strongly recommended for ASR models)
- PyTorch: https://download.pytorch.org/whl/cu124

## Installation
1. Navigate to the project directory:
   ```bash
   cd video-translation
   ```
2. Install base dependencies (excluding PyTorch):
   ```bash
   uv venv
   uv pip install -r requirements.txt
   ```
3. Install the correct PyTorch version for your GPU:

   - **RTX 50-series**:
     ```bash
     uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
     ```
   - **RTX 30/40-series**:
     ```bash
     uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu126
     ```
   - **CPU**:
     ```bash
     pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
     ```


Run the main script with the path to your input video or a folder of videos:

```bash
# Process a single video
uv run main.py path/to/video.mp4 --model qwen

# Process all videos in a folder
uv run main.py path/to/videos/ --model qwen
```

### Options
- `--language`: Target language for translation (default: `Russian`).
- `--model`: (Required) LLM model name for translation. Used to unload the model from VRAM before running ASR.
- `--style`: Path to a text file containing additional style instructions for the translation (e.g., `style.txt`). These instructions are appended to the base translation prompt. You can use the `{language}` placeholder in your text file.
- `--max-words`: Maximum words per subtitle segment (default: `12`).
- `--max-chars`: Maximum characters per subtitle segment (default: `40`).
- `--end-padding`: Extra duration in seconds to keep subtitles on screen after speech ends to improve readability (default: `0.8`). Prevents overlaps with consecutive subtitles.

Example:
```bash
uv run main.py my_video.mp4 --language Spanish --model qwen

# Use a custom style file for translation
uv run main.py my_video.mp4 --model qwen --style style.txt
```

## Local Translation Setup
To translate subtitles, ensure Ollama is running locally. The script communicates directly with the native Ollama endpoint (`http://127.0.0.1:11434/api/chat`).

## Project Structure
- `main.py`: Entrypoint for CLI args and video processing orchestrator.
- `requirements.txt`: Dependencies list.
- `src/`:
  - `audio_extractor.py`: Handles extraction of mono audio via `ffmpeg-python`.
  - `transcriber.py`: Loads Qwen3 ASR and forced-alignment models.
  - `srt_generator.py`: Groups words/tokens into SRT formatted blocks.
  - `translator.py`: Connects to local OpenAI API to translate segments.
  - `video_burner.py`: Re-encodes video with burned subtitles via FFmpeg.
- `tests/`: Pytest test suite.
