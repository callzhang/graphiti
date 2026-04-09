from types import SimpleNamespace

from pydantic import BaseModel

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_base_client import DEFAULT_REASONING
from graphiti_core.llm_client.openai_client import OpenAIClient


def test_default_reasoning_uses_supported_openai_effort() -> None:
    client = OpenAIClient(config=LLMConfig(), client=object())

    assert DEFAULT_REASONING == 'low'
    assert client.reasoning == 'low'


class DummyStructuredResponse(BaseModel):
    foo: str


def test_structured_response_uses_sdk_parsed_payload() -> None:
    client = OpenAIClient(config=LLMConfig(), client=object())

    response = SimpleNamespace(
        output_parsed=DummyStructuredResponse(foo='bar'),
        usage=SimpleNamespace(input_tokens=11, output_tokens=7),
    )

    parsed, input_tokens, output_tokens = client._handle_structured_response(response)

    assert parsed == {'foo': 'bar'}
    assert input_tokens == 11
    assert output_tokens == 7
