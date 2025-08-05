from dataclasses import dataclass, field
from enum import Enum

from src.summarize_algorithms.memory_bank.memory_storage import MemoryStorage


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
    query: str
    current_session_index: int = 0
    response: str = ""

    @property
    def current_context(self) -> Session:
        return self.dialogue_sessions[-1]


@dataclass
class RecsumDialogueState(DialogueState):
    memory: list[list[str]] = field(default_factory=list)

    @property
    def latest_memory(self) -> str:
        return "\n".join(self.memory[-1]) if self.memory else ""


@dataclass
class MemoryBankDialogueState(DialogueState):
    memory_storage: MemoryStorage = field(default_factory=MemoryStorage)


class WorkflowNode(Enum):
    UPDATE_MEMORY = "update_memory"
    GENERATE_RESPONSE = "generate_response"


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"
