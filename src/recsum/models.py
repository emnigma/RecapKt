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

@dataclass
class DialogueState:
    dialogue_sessions: list[Session]
    current_session_index: int
    memory: Optional[list[str]]
    response: str

    @property
    def current_context(self):
        last_session = self.dialogue_sessions[-1]
        if len(last_session) == 0:
            return ""
        return last_session

    @property
    def latest_memory(self):
        if self.memory is None or len(self.memory) == 0:
            return ""
        return self.memory[-1]