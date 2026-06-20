# VRAM Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Free GPU memory (VRAM) held by ASR models prior to running LLM-based translation to prevent GPU out-of-memory issues.

**Architecture:** Implement a `free_model()` utility inside `src/transcriber.py` that clears the global model reference, triggers Python garbage collection, and empties PyTorch's CUDA memory cache. Integrate this function call into `main.py` directly after ASR processing is finished.

**Tech Stack:** Python, PyTorch, gc

---

### Task 1: Add VRAM release function to transcriber

**Files:**
- Modify: `src/transcriber.py`
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: Write the failing test**

Add `test_free_model` to `tests/test_transcriber.py`:
```python
def test_free_model():
    from src.transcriber import free_model
    # Define a dummy model on global namespace for test purposes
    import src.transcriber as transcriber
    transcriber.model = "dummy_model"
    
    free_model()
    
    assert transcriber.model is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcriber.py::test_free_model -v`
Expected: FAIL with `ImportError: cannot import name 'free_model' from 'src.transcriber'`

- [ ] **Step 3: Write minimal implementation**

Add `free_model` to `src/transcriber.py`:
```python
def free_model():
    global model
    if model is not None:
        del model
        model = None
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcriber.py::test_free_model -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/transcriber.py tests/test_transcriber.py
git commit -m "feat: add free_model to release transcriber VRAM"
```

---

### Task 2: Call release function in main pipeline

**Files:**
- Modify: `main.py`
- Test: `tests/test_main.py`

- [ ] **Step 1: Write the failing test**

Modify `tests/test_main.py` to assert that `free_model` is imported and called. We mock `free_model` to verify execution:
```python
from unittest.mock import patch

@patch('main.free_model')
@patch('main.extract_audio')
@patch('main.transcribe_audio')
@patch('main.group_alignments')
@patch('main.burn_subtitles')
@patch('main.generate_srt')
def test_process_video_calls_free_model(mock_gen_srt, mock_burn, mock_group, mock_transcribe, mock_extract, mock_free):
    import main
    from unittest.mock import MagicMock
    mock_transcription = MagicMock()
    mock_transcription.time_stamps = ["dummy"]
    mock_transcribe.return_value = mock_transcription
    
    main.process_video("test.mp4")
    
    mock_free.assert_called_once()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py::test_process_video_calls_free_model -v`
Expected: FAIL with `AttributeError: module 'main' has no attribute 'free_model'` (or failed mock call assertion)

- [ ] **Step 3: Write minimal implementation**

Modify `main.py` to import `free_model` and invoke it after `group_alignments` is complete:
```python
from src.transcriber import transcribe_audio, free_model
```
And in `process_video`:
```python
    segments = group_alignments(transcription.time_stamps)
    
    # Free ASR model VRAM before translation
    free_model()
    
    if do_translate:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py::test_process_video_calls_free_model -v`
Expected: PASS

- [ ] **Step 5: Run all tests to verify whole suite**

Run: `python -m pytest`
Expected: PASS (10/10 tests passed)

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_main.py
git commit -m "feat: call free_model in main pipeline before translation"
```
