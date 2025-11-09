from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
from src.benchmarking.tool_metrics.json_schemas import PLAN_SCHEMA
from src.benchmarking.tool_metrics.prompts import PLAN_PROMPT
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import BaseBlock, MetricState, Session


class Calculator:
    """
    Class for run, evaluate (by compare with reference) and save results of llm's memory algorithms.
    """
    @staticmethod
    def evaluate(
            algorithms: list[Dialog],
            evaluator_functions: list[BaseEvaluator],
            sessions: list[Session],
            query: BaseBlock,
            reference: list[BaseBlock],
            logger: BaseLogger
    ) -> list[dict[str, Any]]:
        """
        The main method for run and evaluate algorithm with the ast sessions.
        :param algorithms: class that implements Dialog protocol.
        :param evaluator_functions: list of class inheritances of BaseEvaluator - for evaluating algorithm's results.
        :param sessions: past user-tool-model interactions.
        :param query: BaseBlock with role 'USER' - the last query for model for comparing results
        :param reference: reference model's response for comparing with algorithm's results.
        :param logger: the class for saving results of running and evaluating algorithms.
        :return: list[dict[str, Any]]: results of running and evaluating algorithm.
        """
        assert query.role == "USER", "query should be BaseBlock with the role 'USER'"

        metrics: list[dict[str, Any]] = []
        for algorithm in algorithms:
            prompt = Calculator.__prepare_query_for_the_first_stage(query)
            state = algorithm.process_dialogue(sessions, prompt, PLAN_SCHEMA)

            algorithm_metrics: list[MetricState] = []
            for evaluator_function in evaluator_functions:
                metric: MetricState = evaluator_function.evaluate(sessions, query, state, reference)
                algorithm_metrics.append(metric)

            record: dict[str, Any] = logger.log_iteration(
                algorithm.__class__.__name__,
                query.content,
                1,
                sessions,
                state,
                algorithm_metrics
            )

            metrics.append(record)

        return metrics

    @staticmethod
    def __prepare_query_for_the_first_stage(query: BaseBlock) -> str:
        return PLAN_PROMPT.format(query.content)
