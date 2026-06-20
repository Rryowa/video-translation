import pytest
from src.transcriber import transcribe_audio

def test_transcribe_audio_callable():
    assert hasattr(transcribe_audio, "__call__")

def test_free_model():
    from src.transcriber import free_model
    # Define a dummy model on global namespace for test purposes
    import src.transcriber as transcriber
    transcriber.model = "dummy_model"
    
    free_model()
    
    assert transcriber.model is None
