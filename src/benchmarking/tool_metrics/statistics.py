import json
import logging
from collections import Counter
from itertools import product
from math import fsum
from pathlib import Path
from typing import Any, TypeVar

from src.benchmarking.base_logger import BaseLogger
from src.benchmarking.models.dtos import StatisticsDto, BaseRecord, AlgorithmStatistics, MemoryRecord
from src.benchmarking.models.enums import MetricType
from src.benchmarking.tool_metrics.calculator import Calculator
from src.benchmarking.tool_metrics.evaluators.base_evaluator import BaseEvaluator
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
                    key = (record.system, metric_state.metric_name)
                    values_by_alg_metric[key].append(float(metric_state.metric_value))

        return Statistics.__get_statistic_metrics(count_of_launches, system_logger, values_by_alg_metric)

    @staticmethod
    def calculate_by_logs(
            count_of_launches: int,
            metrics: list[BaseRecord],
            system_logger: logging.Logger = logging.getLogger()
    ) -> StatisticsDto:
        values_by_alg_metric: dict[tuple[str, MetricType], list[float]] = {}
        for _ in range(count_of_launches):
            for record in metrics:
                if record.metric is None:
                    continue
                for metric_state in record.metric:
                    key = (record.system, metric_state.metric_name)
                    if key not in values_by_alg_metric:
                        values_by_alg_metric[key] = []
                    values_by_alg_metric[key].append(float(metric_state.metric_value))

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

    @staticmethod
    def print_statistics(stats: StatisticsDto) -> None:
        """
        Печать сводной статистики по алгоритмам и метрикам.
        """
        print("=== Statistics ===")
        for alg_stat in stats.algorithms:
            metric_name = alg_stat.metric.value

            print(f"Algorithm: {alg_stat.name}")
            print(f"  Metric: {metric_name}")
            print(f"  Count of launches: {alg_stat.count_of_launches}")
            print(f"  Math. expectation: {alg_stat.mean:.4f}")
            print(f"  Variance: {alg_stat.variance:.4f}")
            print()


if __name__ == "__main__":
    #print(Statistics.calculate_mode([1.0, 1.12, 1.2, 1.3, 3.0, 3.1, 4.0, 5.5, 5.6, 6.0]))
    records: list[BaseRecord] = []
    for f in [
        "BaseMemoryBank",
        "BaseRecsum",
        "FullBaseline",
        "LastBaseline",
        "RagMemoryBank",
        "RagRecsum"
    ]:
        folder = Path("logs/memory/2025-11-15T21:57:10.007355") / f
        for path in folder.glob("*.json"):
            print(path)
            with path.open("r", encoding="utf-8") as f:
                obj = json.load(f)
                if obj.get("memory") is None:
                    record = BaseRecord.from_dict(obj)
                else:
                    record = MemoryRecord.from_dict(obj)
                records.append(record)
    stats = Statistics.calculate_by_logs(
        count_of_launches=10,
        metrics=records
    )
    Statistics.print_statistics(stats)
