# Qwen3-ASR Subtitle Burn Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python script that extracts audio from a video, transcribes it with Qwen3-ASR to get exact word timestamps, generates an SRT file, and burns the subtitles back into the video using FFmpeg.

**Architecture:** A Python script using `qwen-asr` and `ffmpeg-python` to chain media extraction, model inference, subtitle formatting, and subtitle burning.

**Tech Stack:** Python, `qwen-asr`, FFmpeg, `ffmpeg-python`, `torch`

---

### Task 1: Project Setup & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `tests/test_setup.py`

- [ ] **Step 1: Write the failing test**

```python
def test_setup():
    pass
```

- [ ] **Step 2: Write requirements.txt**

```txt
torch
transformers
qwen-asr
ffmpeg-python
pytest
```

- [ ] **Step 3: Run test to verify it passes**

Run: `pytest tests/test_setup.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git init
git add requirements.txt tests/test_setup.py
git commit -m "chore: add dependencies and test setup"
```

### Task 2: Audio Extraction Module

**Files:**
- Create: `src/audio_extractor.py`
- Create: `tests/test_audio_extractor.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import pytest
from src.audio_extractor import extract_audio

def test_extract_audio(tmp_path):
    assert hasattr(extract_audio, "__call__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audio_extractor.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import ffmpeg
import os

def extract_audio(video_path: str, output_audio_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    (
        ffmpeg
        .input(video_path)
        .output(output_audio_path, acodec='pcm_s16le', ac=1, ar='16k')
        .overwrite_output()
        .run(quiet=True)
    )
    return output_audio_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_audio_extractor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/audio_extractor.py tests/test_audio_extractor.py
git commit -m "feat: add audio extraction module"
```

### Task 3: Qwen3-ASR Transcription

**Files:**
- Create: `src/transcriber.py`
- Create: `tests/test_transcriber.py`

- [ ] **Step 1: Write the failing test**

```python
from src.transcriber import transcribe_audio

def test_transcribe_audio_exists():
    assert hasattr(transcribe_audio, "__call__")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_transcriber.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import torch
from qwen_asr import Qwen3ASRModel

model = None

def get_model():
    global model
    if model is None:
        model = Qwen3ASRModel.from_pretrained(
            "Qwen/Qwen3-ASR-1.7B", 
            dtype=torch.bfloat16, 
            device_map="cuda:0"
        )
    return model

def transcribe_audio(audio_path: str):
    m = get_model()
    result = m.transcribe(audio_path)
    return result
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_transcriber.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/transcriber.py tests/test_transcriber.py
git commit -m "feat: add qwen3 asr transcription"
```

### Task 4: SRT Generator

**Files:**
- Create: `src/srt_generator.py`
- Create: `tests/test_srt_generator.py`

- [ ] **Step 1: Write the failing test**

```python
from src.srt_generator import format_time, generate_srt

def test_format_time():
    assert format_time(1.234) == "00:00:01,234"
    assert format_time(3661.5) == "01:01:01,500"

def test_generate_srt(tmp_path):
    segments = [{"start": 0.5, "end": 1.5, "text": "Hello world"}]
    out_file = tmp_path / "out.srt"
    generate_srt(segments, str(out_file))
    content = out_file.read_text()
    assert "00:00:00,500 --> 00:00:01,500" in content
    assert "Hello world" in content
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_srt_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import math

def format_time(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - math.floor(seconds)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def generate_srt(segments: list, output_srt_path: str):
    with open(output_srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = format_time(seg['start'])
            end = format_time(seg['end'])
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{seg['text'].strip()}\n\n")
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_srt_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/srt_generator.py tests/test_srt_generator.py
git commit -m "feat: add srt generator"
```

### Task 5: Video Subtitle Burner

**Files:**
- Create: `src/video_burner.py`
- Create: `tests/test_video_burner.py`

- [ ] **Step 1: Write the failing test**

```python
from src.video_burner import burn_subtitles

def test_burn_subtitles_exists():
    assert hasattr(burn_subtitles, "__call__")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_video_burner.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import ffmpeg
import os

def burn_subtitles(video_path: str, srt_path: str, output_video_path: str):
    escaped_srt = srt_path.replace('\\', '\\\\').replace(':', '\\:')
    
    (
        ffmpeg
        .input(video_path)
        .output(output_video_path, vf=f"subtitles='{escaped_srt}'")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_video_path
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_video_burner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
```bash
git add src/video_burner.py tests/test_video_burner.py
git commit -m "feat: add subtitle burning module"
```

### Task 6: Main Orchestration Script

**Files:**
- Create: `main.py`
- Create: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

```python
from main import process_video

def test_process_video_exists():
    assert hasattr(process_video, "__call__")
```

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_main.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
import argparse
import os
from src.audio_extractor import extract_audio
from src.transcriber import transcribe_audio
from src.srt_generator import generate_srt
from src.video_burner import burn_subtitles

def process_video(video_path: str):
    base_name = os.path.splitext(video_path)[0]
    audio_path = f"{base_name}.wav"
    srt_path = f"{base_name}.srt"
    output_video = f"{base_name}_subtitled.mp4"
    
    extract_audio(video_path, audio_path)
    segments = transcribe_audio(audio_path)
    generate_srt(segments, srt_path)
    burn_subtitles(video_path, srt_path, output_video)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Burn Qwen3-ASR subtitles into video")
    parser.add_argument("video", help="Path to input video")
    args = parser.parse_args()
    process_video(args.video)
```

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_main.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: main orchestration script"
```
