from typing import Protocol

from src.summarize_algorithms.core.models import DialogueState, Session


class Dialog(Protocol):
    def process_dialogue(self, sessions: list[Session], query: str) -> DialogueState:
        ...
