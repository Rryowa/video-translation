import pytest
from main import process_video
from unittest.mock import patch

def test_process_video_callable():
    assert hasattr(process_video, "__call__")

@patch('main.translate_segments')
@patch('main.free_model')
@patch('main.extract_audio')
@patch('main.transcribe_audio')
@patch('main.group_alignments')
@patch('main.burn_subtitles')
@patch('main.generate_srt')
def test_process_video_calls_free_model(mock_gen_srt, mock_burn, mock_group, mock_transcribe, mock_extract, mock_free, mock_translate):
    import main
    from unittest.mock import MagicMock
    mock_transcription = MagicMock()
    mock_transcription.time_stamps = ["dummy"]
    mock_transcribe.return_value = mock_transcription
    
    main.process_video("test.mp4")
    
    mock_free.assert_called_once()
    mock_translate.assert_called_once()
