from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_base_client import DEFAULT_REASONING
from graphiti_core.llm_client.openai_client import OpenAIClient


def test_default_reasoning_uses_supported_openai_effort() -> None:
    client = OpenAIClient(config=LLMConfig(), client=object())

    assert DEFAULT_REASONING == 'low'
    assert client.reasoning == 'low'
