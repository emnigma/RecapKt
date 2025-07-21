from dataclasses import dataclass
from enum import Enum
from typing import TypedDict


class Role(Enum):
    USER = "user"
    BOT = "bot"


@dataclass
class Message:
    role: Role
    message: str

    def __str__(self) -> str:
        return f"{self.role.value}: {self.message}"


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
    memory: str
    response: str
