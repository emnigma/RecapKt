from typing import Any, Protocol

from src.summarize_algorithms.core.models import DialogueState, Session


class Dialog(Protocol):
    system_name: str

    def process_dialogue(
            self,
            sessions: list[Session],
            query: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> DialogueState:
        ...
