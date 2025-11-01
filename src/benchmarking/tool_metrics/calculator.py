from pathlib import Path
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.tool_metrics.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import DialogueState, MetricState, Session


class Calculator:
    """
    Class for run, evaluate (by compare with reference) and save results of llm's memory algorithms.
    """

    def __init__(self, logger: BaseLogger, path_to_save: str | Path | None = None) -> None:
        """
        Initialization of Calculator class.
        :param logger: the class for saving running and evaluating results.
        :param path_to_save: path to save the result.
        """
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
        """
        The main method for run and evaluate algorithm with the ast sessions.
        :param algorithms: class that implements Dialog protocol.
        :param evaluator_function: class inheritance of BaseEvaluator - for evaluating algorithm's results.
        :param sessions: past user-tool-model interactions.
        :param reference_session: reference session for comparing with algorithm's results.
        :param count_of_sessions_to_evaluate: shows how many past interactions sessions should be used for memory.
        :return: list[dict[str, Any]]: results of running and evaluating algorithm.
        """
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
