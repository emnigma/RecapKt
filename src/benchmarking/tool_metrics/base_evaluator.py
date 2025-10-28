from abc import ABC, abstractmethod

from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    MetricState,
    Session,
)


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(
            self,
            sessions: list[Session],
            query: BaseBlock,
            state: DialogueState | str,
            reference: BaseBlock | None = None
    ) -> MetricState:
        ...
