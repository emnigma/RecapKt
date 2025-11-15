from collections import Counter
from math import fsum
from pathlib import Path
from typing import Any, TypeVar
from itertools import product

import logging

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.calculator import Calculator
from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
from src.benchmarking.models.dtos import StatisticsDto, BaseRecord, AlgorithmStatistics
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import Session, BaseBlock


NumberT = TypeVar("NumberT", int, float)


class Statistics:
    @staticmethod
    def calculate(
            count_of_launches: int,
            algorithms: list[Dialog],
            evaluator_functions: list[BaseEvaluator],
            sessions: list[Session],
            prompt: str,
            reference: list[BaseBlock],
            logger: BaseLogger,
            tools: list[dict[str, Any]] | None = None,
            subdirectory: str | Path | None = None,
            shuffle: bool = False
    ) -> StatisticsDto:
        system_logger = logging.getLogger()

        if shuffle:
            prepared_sessions: list[list[Session]] = [list(s) for s in product(sessions)]
            system_logger.info("Sessions shuffled")
        else:
            prepared_sessions: list[list[Session]] = [sessions] * count_of_launches

        assert count_of_launches <= len(prepared_sessions),\
            "Count of launches should be less or equal than square of len(sessions)."

        values_by_alg_metric: dict[tuple[str, MetricType], list[float]] = {}
        for i in range(count_of_launches):
            system_logger.info("Starting evaluation")
            #TODO дописать логгирование итераций
            metrics: list[BaseRecord] = Calculator.evaluate(
                algorithms,
                evaluator_functions,
                prepared_sessions[i],
                prompt,
                reference,
                logger,
                tools,
                subdirectory
            )

            for record in metrics:
                if record.metrics is None:
                    continue
                for metric_state in record.metrics:
                    key = (record.system, metric_state.metric)
                    values_by_alg_metric[key].append(float(metric_state.value))

        return Statistics.__get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric)

    @staticmethod
    def calculate_by_logs(
            count_of_launches: int,
            log_records: list[BaseRecord]
    ) -> StatisticsDto:
        system_logger = logging.getLogger()

        values_by_alg_metric: dict[tuple[str, MetricType], list[float]] = {}
        for record in log_records:
            if record.metrics is None:
                continue
            for metric_state in record.metrics:
                key = (record.system, metric_state.metric)
                values_by_alg_metric.setdefault(key, []).append(float(metric_state.value))

        return Statistics.__get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric)

    @staticmethod
    def __get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric):
        algorithm_stats: list[AlgorithmStatistics] = []
        system_logger.info("Getting statistics...")
        for (alg_name, metric_type), values in values_by_alg_metric.items():
            system_logger.info(f"Getting {alg_name} {metric_type.value} statistics")
            n = len(values)
            if n == 0:
                continue

            mean = fsum(values) / n

            variance = fsum((v - mean) ** 2 for v in values) / n

            algorithm_stats.append(
                AlgorithmStatistics(
                    name=alg_name,
                    metric=metric_type,
                    count_of_launches=count_of_launches,
                    mean=mean,
                    variance=variance
                )
            )
        return StatisticsDto(
            algorithms=algorithm_stats
        )

    @staticmethod
    def calculate_mode[NumberT](values: list[NumberT]) -> NumberT:
        if isinstance(NumberT, int):
            counter = Counter(values)
            mode, _ = counter.most_common(1)[0]
            return mode

        values.sort()
        left: float = values[0]
        right: float = values[2]
        last_group: list[float] = []
        max_group: list[float] = []
        for i in range(1, len(values) - 2):
            if abs(values[i] - left) <= abs(values[i] - right):
                last_group.append(values[i])
            elif abs(values[i] - left) > abs(values[i] - right):
                if len(last_group) > len(max_group):
                    max_group = last_group
                last_group = [values[i]]

        return fsum(max_group) / len(max_group)


if __name__ == "__main__":
    print(Statistics.calculate_mode([1.0, 1.12, 1.2, 1.3, 3.0, 3.1, 4.0, 5.5, 5.6, 6.0]))
