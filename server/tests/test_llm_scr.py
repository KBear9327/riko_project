import os
import json
import pytest
from unittest.mock import MagicMock, patch
from server.process.llm_funcs.llm_scr import llm_response, load_history, save_history, SYSTEM_PROMPT

HISTORY_FILE = "test_history.json"

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    monkeypatch.setattr('server.process.llm_funcs.llm_scr.HISTORY_FILE', HISTORY_FILE)
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.output_text = "Test response"
    mock_client.responses.create.return_value = mock_response
    monkeypatch.setattr('server.process.llm_funcs.llm_scr.client', mock_client)

@pytest.fixture
def clean_history():
    # Ensure the history file is clean before each test
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
    yield
    # Clean up the history file after each test
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)

def test_load_history_no_file():
    # Test loading history when the file doesn't exist
    assert load_history() == SYSTEM_PROMPT

def test_save_and_load_history(clean_history):
    # Test saving and then loading history
    history = SYSTEM_PROMPT + [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}]
    save_history(history)
    assert load_history() == history

def test_llm_response(clean_history):
    # Test the main llm_response function
    user_input = "Hello, world!"
    response = llm_response(user_input)

    # Check that the response is correct
    assert response == "Test response"

    # Check that the history was updated correctly
    history = load_history()
    assert len(history) == 3 # System prompt, user message, assistant message
    assert history[1]["role"] == "user"
    assert history[1]["content"][0]["text"] == user_input
    assert history[2]["role"] == "assistant"
    assert history[2]["content"][0]["text"] == "Test response"
