import logging
import os

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.summarize_algorithms.core.models import (
    DialogueState,
    MemoryBankDialogueState,
    MetricState,
    RecsumDialogueState,
    Session,
)


class BaseLogger(ABC):
    def __init__(self, logs_dir: str | Path = "logs/memory") -> None:
        os.makedirs(logs_dir, exist_ok=True)
        self.log_dir = Path(logs_dir)
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState | None,
            metric: MetricState | None = None,
            is_return: bool = False
    ) -> None | dict[str, Any]:
        ...

    @staticmethod
    def _serialize_memories(state: DialogueState) -> dict[str, Any]:
        result: dict[str, Any] = {}

        if state.code_memory_storage is not None:
            result["code_memory_storage"] = state.code_memory_storage.to_dict()
        if state.tool_memory_storage is not None:
            result["tool_memory_storage"] = state.tool_memory_storage.to_dict()
        if isinstance(state, MemoryBankDialogueState):
            result["text_memory_storage"] = state.text_memory_storage.to_dict()
        if isinstance(state, RecsumDialogueState):
            result["text_memory"] = state.text_memory

        return result
