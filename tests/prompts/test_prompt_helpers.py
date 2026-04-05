from __future__ import annotations

from graphiti_core.prompts.prompt_helpers import to_prompt_json


class FakeDateTime:
    def iso_format(self) -> str:
        return "2026-04-02T12:00:00+00:00"


class FakeModel:
    def model_dump(self) -> dict[str, object]:
        return {"created_at": FakeDateTime(), "name": "Atlas"}


def test_to_prompt_json_normalizes_datetime_like_and_model_values() -> None:
    rendered = to_prompt_json(
        {
            "existing_nodes": [
                {"name": "Atlas", "created_at": FakeDateTime()},
                FakeModel(),
            ]
        }
    )

    assert "2026-04-02T12:00:00+00:00" in rendered
    assert '"Atlas"' in rendered
