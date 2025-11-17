import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.benchmarking.baseline import DialogueBaseline
from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.tool_metrics.evaluators.f1_tool_evaluator import F1ToolEvaluator
from src.benchmarking.tool_metrics.load_session import Loader
from src.benchmarking.models.dtos import QueryAndReference, StatisticsDto, BaseRecord, MemoryRecord
from src.benchmarking.tool_metrics.statistics import Statistics
from src.summarize_algorithms.core.models import BaseBlock, Session
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem


class Runner:
    def __init__(self, templates_dir: str = "prompts") -> None:
        self.logger = logging.getLogger()
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=True,
            trim_blocks=True
        )

    def run(self, dir_path: Path, session_file: Path | str) -> None:
        memory_logger = MemoryLogger()
        baseline_logger = BaselineLogger()

        base_recsum = RecsumDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseRecsum")
        base_memory_bank = MemoryBankDialogueSystem(embed_code=False, embed_tool=False, system_name="BaseMemoryBank")
        rag_recsum = RecsumDialogueSystem(embed_code=True, embed_tool=True, system_name="RagRecsum")
        rag_memory_bank = MemoryBankDialogueSystem(embed_code=True, embed_tool=True, system_name="RagMemoryBank")
        full_baseline = DialogueBaseline("FullBaseline")
        last_baseline = DialogueBaseline("LastBaseline")

        algorithms_with_memory = [
            base_recsum,
            base_memory_bank,
            rag_recsum,
            rag_memory_bank
        ]
        baseline_algorithms = [
            full_baseline,
            last_baseline
        ]

        self.logger.info("Start parsing session")
        past_interactions = Loader.load_session(dir_path / session_file)

        query_and_reference = self.__execute_query_and_reference(past_interactions)
        query, reference = query_and_reference.query, query_and_reference.reference
        prompt = self.__prepare_query_for_the_first_stage(query)

        f1_tool_evaluator = F1ToolEvaluator()

        subdirectory: Path = Path(datetime.now().isoformat())

        self.logger.info("Start evaluating baseline statistics")
        baseline_statistics: StatisticsDto = Statistics.calculate(
            10,
            baseline_algorithms,
            [f1_tool_evaluator],
            [past_interactions],
            prompt,
            reference,
            baseline_logger,
            None,
            subdirectory
        )

        self.logger.info("Start evaluating memory statistics")
        memory_statistics: StatisticsDto = Statistics.calculate(
            10,
            algorithms_with_memory,
            [f1_tool_evaluator],
            [past_interactions],
            prompt,
            reference,
            memory_logger,
            None,
            subdirectory
        )

        Runner.__print_statistics(
            StatisticsDto(algorithms=[*baseline_statistics.algorithms, *memory_statistics.algorithms])
        )

    @staticmethod
    def get_statistics_by_directory_with_logs(path: Path | str) -> StatisticsDto:
        records: list[BaseRecord] = []
        for f in [
            "BaseMemoryBank",
            "BaseRecsum",
            "FullBaseline",
            "LastBaseline",
            "RagMemoryBank",
            "RagRecsum"
        ]:
            folder = Path(path) / f
            for path in folder.glob("*.json"):
                print(path)
                with path.open("r", encoding="utf-8") as f:
                    obj = json.load(f)
                    if obj.get("memory") is None:
                        record = BaseRecord.from_dict(obj)
                    else:
                        record = MemoryRecord.from_dict(obj)
                    records.append(record)
        return Statistics.calculate_by_logs(
            count_of_launches=10,
            metrics=records
        )

    def __execute_query_and_reference(self, past_interactions: Session) -> QueryAndReference:
        reference: list[BaseBlock] = []
        query: BaseBlock | None = None
        for i in range(len(past_interactions.messages) - 1, -1, -1):
            if past_interactions.messages[i].role == "USER" and past_interactions.messages[i].content != "":
                self.logger.info(f"User founded {i}")
                self.logger.info(f"User message: {past_interactions.messages[i].content}")
                query = past_interactions.messages[i]
                break
            else:
                reference.append(past_interactions.messages[i])

        assert query is not None, "User's query is not founded."

        reference = reference[::-1]

        return QueryAndReference(
            query=query,
            reference=reference
        )

    def __prepare_query_for_the_first_stage(self, query: BaseBlock) -> str:
        template = self.env.get_template("first_stage.j2")
        rendered_prompt = template.render(query=query.content)
        return rendered_prompt

    @staticmethod
    def __print_metrics(log_records: list[dict[str, Any]]) -> None:
        for record in log_records:
            print(f"System: {record['system']}")
            print("Metrics:")
            for metric in record.get("metric"):
                print(f"  - {metric.get("metric_name")}: {metric.get("metric_value")}")
            print("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    Statistics.print_statistics(
        Runner.get_statistics_by_directory_with_logs("logs/memory/2025-11-15T21:57:10.007355")
    )

    runner = Runner()
    runner.run(
        dir_path=Path("/Users/mikhailkharlamov/Documents/.../create-agents-md"),
        session_file="chat_20251022_001157_763_6010.messages.json"
    )
