from pathlib import Path
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.tool_metrics.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import MetricState, Session


class Calculator:
    """
    Class for run, evaluate (by compare with reference) and save results of llm's memory algorithms.
    """

    def __init__(self, logger: BaseLogger, path_to_save: str | Path | None = None) -> None:
        """
        Initialization of Calculator class.
        :param logger: the class for saving results of running and evaluating algorithms.
        :param path_to_save:
        """
        self.path_to_save: str | Path | None = path_to_save
        self.logger = logger

    def evaluate(
            self,
            algorithms: list[Dialog],
            evaluator_function: BaseEvaluator,
            sessions: list[Session],
            reference_session: Session
    ) -> list[dict[str, Any]]:
        """
        The main method for run and evaluate algorithm with the ast sessions.
        :param algorithms: class that implements Dialog protocol.
        :param evaluator_function: class inheritance of BaseEvaluator - for evaluating algorithm's results.
        :param sessions: past user-tool-model interactions.
        :param reference_session: reference session for comparing with algorithm's results.
        :return: list[dict[str, Any]]: results of running and evaluating algorithm.
        """
        user_role_messages = reference_session.get_messages_by_role("USER")
        query = user_role_messages[-1]

        metrics: list[dict[str, Any]] = []
        for algorithm in algorithms:
            state = algorithm.process_dialogue(sessions, query.content)
            metric: MetricState = evaluator_function.evaluate(sessions, query, state)

            record: dict[str, Any] | None = self.logger.log_iteration(
                algorithm.__class__.__name__,
                query.content,
                1,
                sessions,
                state,
                metric
            )

            assert record is not None, "logger didn't return a value"

            metrics.append(record)

        return metrics
