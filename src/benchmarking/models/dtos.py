from dataclasses import dataclass, field
from typing import Any

from dataclasses_json import dataclass_json

from src.benchmarking.models.enums import MetricType, ModelPriceForMillionTokensInDollars
from src.summarize_algorithms.core.models import BaseBlock, OpenAIModels


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
    metric_name: MetricType
    metric_value: float | int


@dataclass_json
@dataclass
class BaseRecord:
    timestamp: str
    iteration: int
    system: str
    query: str
    response: Any
    sessions: list[dict[str, Any]]
    metric: list[MetricState] | None = field(default=None)


@dataclass_json
@dataclass
class MemoryRecord(BaseRecord):
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass_json
@dataclass
class Evaluation:
    memory: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenInfo:
    model: OpenAIModels
    price: ModelPriceForMillionTokensInDollars
    input_tokens: int
    output_tokens: int
    input_price: float
    output_price: float
    full_price: float
