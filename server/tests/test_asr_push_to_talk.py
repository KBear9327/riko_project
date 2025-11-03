import sys
from unittest.mock import MagicMock
# Mock the sounddevice module before it's imported by the code under test
sys.modules['sounddevice'] = MagicMock()

import os
import pytest
from server.process.asr_func.asr_push_to_talk import transcribe_audio
import numpy as np
import soundfile as sf

@pytest.fixture
def mock_whisper_model():
    model = MagicMock()
    # Mock the transcribe method to return a predictable result
    model.transcribe.return_value = ([MagicMock(text="Hello world")], None)
    return model

@pytest.fixture
def audio_file(tmp_path):
    # Create a dummy audio file for testing
    filepath = tmp_path / "test.wav"
    samplerate = 16000
    data = np.random.uniform(-1, 1, samplerate) # 1 second of random audio
    sf.write(filepath, data, samplerate)
    return str(filepath)

def test_transcribe_audio(mock_whisper_model, audio_file):
    # Call the function with the mocked model and the dummy audio file
    result = transcribe_audio(mock_whisper_model, audio_file)

    # Assert that the transcribe method was called with the correct audio file
    mock_whisper_model.transcribe.assert_called_once_with(audio_file)

    # Assert that the function returns the expected transcription
    assert result == "Hello world"
