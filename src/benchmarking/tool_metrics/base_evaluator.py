from abc import ABC, abstractmethod

from src.summarize_algorithms.core.models import (
    BaseBlock,
    DialogueState,
    MetricState,
    Session,
)


class BaseEvaluator(ABC):
    """
    Abstract base class for functions that evaluates llm's memory algorithms.
    """

    @abstractmethod
    def evaluate(
            self,
            sessions: list[Session],
            query: BaseBlock,
            state: DialogueState | str,
            reference: BaseBlock | None = None
    ) -> MetricState:
        """
        Returns eval score of llm's answer with for query.
        :param sessions: last user's sessions.
        :param query: the last user's query which response is evaluating.
        :param state: answer of model.
        :param reference: reference model's answer.
        :return: MetricState: dataclass with information about evaluation (metric's name and score).
        """
        ...
