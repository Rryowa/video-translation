import pytest
from main import process_video
from unittest.mock import patch

def test_process_video_callable():
    assert hasattr(process_video, "__call__")

@patch('src.translator.translate_segments')
@patch('src.transcriber.free_model')
@patch('src.audio_extractor.extract_audio')
@patch('src.transcriber.transcribe_audio')
@patch('src.srt_generator.group_alignments')
@patch('src.video_burner.burn_subtitles')
@patch('src.srt_generator.generate_srt')
def test_process_video_calls_free_model(mock_gen_srt, mock_burn, mock_group, mock_transcribe, mock_extract, mock_free, mock_translate):
    import main
    from unittest.mock import MagicMock
    mock_transcription = MagicMock()
    mock_transcription.time_stamps = ["dummy"]
    mock_transcribe.return_value = mock_transcription
    
    main.process_video("test.mp4", "Russian", "qwen")
    
    mock_free.assert_called_once()
    mock_translate.assert_called_once()
