import sys
from unittest.mock import MagicMock
# Mock the sounddevice module before it's imported by the code under test
sys.modules['sounddevice'] = MagicMock()

import pytest
from unittest.mock import patch
from server.process.tts_func.sovits_ping import sovits_gen, play_audio
import soundfile as sf
import numpy as np

@patch('requests.post')
def test_sovits_gen_success(mock_post, tmp_path):
    # Mock the response from the server
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Create a dummy wav file content
    dummy_wav_content = b'RIFF'
    mock_response.content = dummy_wav_content
    mock_post.return_value = mock_response

    output_path = tmp_path / "output.wav"
    result = sovits_gen("Hello", str(output_path))

    # Check that the request was made correctly
    mock_post.assert_called_once()
    assert result == str(output_path)
    # Check that the file was written
    with open(output_path, 'rb') as f:
        assert f.read() == dummy_wav_content

@patch('requests.post')
def test_sovits_gen_error(mock_post):
    # Mock a failed response from the server
    mock_post.side_effect = Exception("Test error")

    result = sovits_gen("Hello")

    assert result is None

@patch('soundfile.read')
def test_play_audio(mock_read):
    # Mock the audio file
    dummy_data = np.random.uniform(-1, 1, 16000)
    mock_read.return_value = (dummy_data, 16000)

    play_audio("dummy_path.wav")

    # Check that the audio was played
    sys.modules['sounddevice'].play.assert_called_once()
    sys.modules['sounddevice'].wait.assert_called_once()
