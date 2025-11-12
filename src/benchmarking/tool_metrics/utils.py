from dataclasses import dataclass

from src.summarize_algorithms.core.models import BaseBlock


@dataclass
class QueryAndReference:
    query: BaseBlock
    reference: list[BaseBlock]
