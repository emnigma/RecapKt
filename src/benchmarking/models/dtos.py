from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import dataclass_json

from src.benchmarking.models.enums import MetricType
from src.summarize_algorithms.core.models import BaseBlock


@dataclass
class QueryAndReference:
    query: BaseBlock
    reference: list[BaseBlock]


@dataclass
class AlgorithmStatistics:
    name: str
    metric: MetricType
    count_of_launches: int
    mean: float
    variance: float
    #mode: int | float


@dataclass
class StatisticsDto:
    algorithms: list[AlgorithmStatistics]


@dataclass_json
@dataclass
class MetricState:
    metric: MetricType
    value: float | int


@dataclass_json
@dataclass
class BaseRecord:
    timestamp: str
    iteration: int
    system: str
    query: str
    response: Any
    sessions: list[dict[str, Any]]
    metrics: list[MetricState] | None = field(default=None)


@dataclass_json
@dataclass
class MemoryRecord(BaseRecord):
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class Evaluation:
    memory: dict[str, Any] = field(default_factory=dict)
