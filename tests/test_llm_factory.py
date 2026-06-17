import sys
from unittest.mock import MagicMock, patch

import pytest


def test_factory_raises_for_unknown_provider():
    with patch.dict("os.environ", {"LLM_PROVIDER": "unknown_xyz"}):
        from llm.factory import get_llm

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm()


def test_factory_anthropic_calls_correct_class():
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_module.ChatAnthropic.return_value = mock_instance

    with patch.dict("os.environ", {"LLM_PROVIDER": "anthropic", "LLM_MODEL": "claude-test", "ANTHROPIC_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"langchain_anthropic": mock_module}):
            from llm import factory
            result = factory.get_llm()

    mock_module.ChatAnthropic.assert_called_once_with(model="claude-test")
    assert result is mock_instance


def test_factory_openai_calls_correct_class():
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_module.ChatOpenAI.return_value = mock_instance

    with patch.dict("os.environ", {"LLM_PROVIDER": "openai", "LLM_MODEL": "gpt-4o", "OPENAI_API_KEY": "test-key"}):
        with patch.dict(sys.modules, {"langchain_openai": mock_module}):
            from llm import factory
            result = factory.get_llm()

    mock_module.ChatOpenAI.assert_called_once_with(model="gpt-4o")
    assert result is mock_instance


def test_factory_ollama_calls_correct_class():
    mock_module = MagicMock()
    mock_instance = MagicMock()
    mock_module.ChatOllama.return_value = mock_instance

    with patch.dict("os.environ", {"LLM_PROVIDER": "ollama", "LLM_MODEL": "llama3"}):
        with patch.dict(sys.modules, {"langchain_ollama": mock_module}):
            from llm import factory
            result = factory.get_llm()

    mock_module.ChatOllama.assert_called_once_with(model="llama3")
    assert result is mock_instance


def test_factory_default_provider_is_anthropic():
    mock_module = MagicMock()
    mock_module.ChatAnthropic.return_value = MagicMock()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}, clear=True):
        with patch.dict(sys.modules, {"langchain_anthropic": mock_module}):
            from llm import factory
            factory.get_llm()

    mock_module.ChatAnthropic.assert_called_once()
