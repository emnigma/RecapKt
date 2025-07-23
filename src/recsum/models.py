from dataclasses import dataclass
from typing import Optional, TypedDict


@dataclass
class Message:
    role: str
    message: str

    def __str__(self) -> str:
        return f"{self.role}: {self.message}"


@dataclass
class Session:
    messages: list[Message]

    def __len__(self) -> int:
        return len(self.messages)

    def __str__(self) -> str:
        return "\n".join([str(message) for message in self.messages])


class DialogueState(TypedDict):
    dialogue_sessions: list[Session]
    current_session_index: int
    memory: Optional[list[str]]
    response: str
