import json
from typing import Any

from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
from src.summarize_algorithms.core.models import Session, DialogueState, BaseBlock, ToolCallBlock
from src.benchmarking.models.dtos import MetricState


class F1ToolEvaluator(BaseEvaluator):
    def evaluate(
            self,
            sessions: list[Session],
            query: str,
            state: DialogueState,
            reference: list[BaseBlock] | None = None
    ) -> MetricState:
        assert reference is not None, "Reference is required for F1 Tool evaluation."
        assert not isinstance(state.response, str), "State response must not be a string."

        reference_tools: set[str] = set(
            map(
                lambda x: x.name,
                filter(
                    lambda x: isinstance(x, ToolCallBlock),
                    reference
                )
            )
        )

        predicted_tools: set[str] = set(
            map(
                lambda x: x.get("name", ""),
                filter(
                    lambda x: x.get("kind", "") == "tool_call" and any(
                        [
                            F1ToolEvaluator.__compare_arguments_for_null(x.get("args", {}), json.loads(r.arguments))
                            for r in reference
                            if isinstance(r, ToolCallBlock)
                        ]
                    ),
                    state.response.get("plan_steps", [])
                )
            )
        )

        true_positives = len(predicted_tools.intersection(reference_tools))
        false_positives = len(predicted_tools.difference(reference_tools))
        false_negatives = len(reference_tools.difference(predicted_tools))

        f1_score = F1ToolEvaluator.__calculate_f1(true_positives, false_positives, false_negatives)

        return MetricState(
            metric="F1_TOOL",
            value=f1_score
        )

    @staticmethod
    def __calculate_f1(true_positives: int, false_positives: int, false_negatives: int) -> float:
        if true_positives + false_positives == 0 or true_positives + false_negatives == 0:
            return 0.0

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        if precision + recall == 0:
            return 0.0

        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score

    @staticmethod
    def __compare_arguments_for_null(first_argument: dict[str, Any], second_argument: dict[str, Any]) -> bool:
        """
        Compares the positions of null arguments in two sets of tool call arguments.
        :return: bool: True if null positions are equal in both arguments, False otherwise.
        """
        first_null_positions = {key for key, value in first_argument.items() if value is None}
        second_null_positions = {key for key, value in second_argument.items() if value is None}
        return first_null_positions == second_null_positions
