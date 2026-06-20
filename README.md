# Video Translation and Subtitling Tool

Automatically extract audio, transcribe with word-level alignment, translate to Russian, and burn subtitles back into a video.

## Features
- **Audio Extraction**: Extracts PCM 16kHz mono audio from video files.
- **ASR & Alignment**: Transcribes audio and forces alignment using Qwen3-ASR (1.7B) and Qwen3-ForcedAligner (0.6B) models.
- **Auto Segmenting**: Intelligently groups aligned tokens into readable subtitle segments based on duration gaps and character limits.
- **Local Translation**: Optional translation to Russian via a local OpenAI-compatible API (e.g., `llama.cpp` or `Ollama`) with customizable reasoning budget.
- **Subtitle Burning**: Renders and burns the final subtitles (SRT) back into the output MP4 video.

## Requirements
- Python 3.10+
- FFmpeg installed and available in system PATH
- CUDA-compatible GPU (strongly recommended for ASR models)

## Installation
1. Navigate to the project directory:
   ```bash
   cd video-translation
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Local Translation Setup
To translate subtitles to Russian, start your local LLM (e.g. llama.cpp) supporting OpenAI API compatibility on `http://127.0.0.1:11434/v1`:

```bash
# Example running llama-server
llama-server -m Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf --port 11434
```

## Usage
Run the main script with the path to your input video:

```bash
python main.py path/to/video.mp4
```

### Options
- `--translate`: Translate generated subtitles to Russian.
- `--translation-model`: The name of your local LLM model (default: `Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q8_0.gguf`).
- `--thinking-budget`: Number of reasoning tokens allocated (default: `500`).

Example:
```bash
python main.py my_video.mp4 --translate --thinking-budget 800
```

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
