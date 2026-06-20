import pytest
from src.transcriber import transcribe_audio

def test_transcribe_audio_callable():
    assert hasattr(transcribe_audio, "__call__")
