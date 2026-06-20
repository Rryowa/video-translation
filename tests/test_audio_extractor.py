import os
import pytest
from src.audio_extractor import extract_audio

def test_extract_audio_callable():
    assert hasattr(extract_audio, "__call__")
