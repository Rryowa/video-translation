import pytest
from src.video_burner import burn_subtitles

def test_burn_subtitles_callable():
    assert hasattr(burn_subtitles, "__call__")
