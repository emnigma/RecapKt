from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Optional


@dataclass
class BaseBlock:
    role: str
    content: str


@dataclass
class CodeBlock(BaseBlock):
    code: str


@dataclass
class ToolCall(BaseBlock):
    id: str
    name: str
    arguments: str
    response: str


class Session:
    def __init__(self, messages: list[BaseBlock]) -> None:
        self.messages = messages

    def __len__(self) -> int:
        return len(self.messages)

    def __getitem__(self, index: int) -> BaseBlock:
        return self.messages[index]

    def __iter__(self) -> Iterator[BaseBlock]:
        return iter(self.messages)

    def __str__(self) -> str:
        return "\n".join(f"{msg.role}: {msg.content}" for msg in self.messages)

    def get_messages_by_role(self, role: str) -> list[BaseBlock]:
        return [msg for msg in self.messages if msg.role == role]

    def get_code_blocks(self) -> list[CodeBlock]:
        return [msg for msg in self.messages if isinstance(msg, CodeBlock)]

    def get_tool_calls(self) -> list[ToolCall]:
        return [msg for msg in self.messages if isinstance(msg, ToolCall)]


@dataclass
class DialogueState:
    dialogue_sessions: list[Session]
    query: str
    current_session_index: int = 0
    _response: Optional[str] = None

    @property
    def response(self) -> str:
        if self._response is None:
            raise ValueError("Response has not been generated yet.")
        return self._response

    @property
    def current_context(self) -> Session:
        return self.dialogue_sessions[-1]


@dataclass
class RecsumDialogueState(DialogueState):
    memory: list[list[str]] = field(default_factory=list)

    @property
    def latest_memory(self) -> str:
        return "\n".join(self.memory[-1]) if self.memory else ""


class WorkflowNode(Enum):
    UPDATE_MEMORY = "update_memory"
    GENERATE_RESPONSE = "generate_response"


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"
