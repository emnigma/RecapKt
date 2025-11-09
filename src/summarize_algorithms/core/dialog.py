from typing import Any, Protocol

from src.summarize_algorithms.core.models import DialogueState, Session


class Dialog(Protocol):
    def process_dialogue(
            self,
            sessions: list[Session],
            query: str,
            structure: dict[str, Any] | None = None,
            tools: dict[str, Any] | None = None
    ) -> DialogueState:
        ...
