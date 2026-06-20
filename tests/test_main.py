import pytest
from main import process_video

def test_process_video_callable():
    assert hasattr(process_video, "__call__")
