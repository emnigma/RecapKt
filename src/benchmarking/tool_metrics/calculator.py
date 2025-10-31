from pathlib import Path
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.tool_metrics.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import DialogueState, MetricState, Session


class Calculator:
    def __init__(self, logger: BaseLogger, path_to_save: str | Path | None = None) -> None:
        self.path_to_save: str | Path | None = path_to_save
        self.logger = logger

    def evaluate(
            self,
            algorithms: list[Dialog],
            evaluator_function: BaseEvaluator,
            sessions: list[Session],
            reference_session: Session,
            count_of_sessions_to_evaluate: int = 0,
    ) -> list[dict[str, Any]]:
        assert count_of_sessions_to_evaluate < len(sessions),\
            "count_of_sessions_to_evaluate is greater then count of entire sessions."

        using_sessions: list[Session] = sessions[:count_of_sessions_to_evaluate]
        user_role_messages = reference_session.get_messages_by_role("USER")
        query = user_role_messages[-1]

        metrics: list[dict[str, Any]] = []
        for algorithm in algorithms:
            state = algorithm.process_dialogue(sessions, str(query))
            metric: MetricState = evaluator_function.evaluate(sessions, query, state)

            record: dict[str, Any] | None = self.logger.log_iteration(
                algorithm.__class__.__name__,
                str(query),
                1,
                using_sessions,
                state if isinstance(state, DialogueState) else None,
                metric,
                True
            )

            assert record is not None, "logger didn't return a value"

            metrics.append(record)

        return metrics
