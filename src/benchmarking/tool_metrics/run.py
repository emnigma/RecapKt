import logging
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.benchmarking.baseline import DialogueBaseline
from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.tool_metrics.calculator import Calculator
from src.benchmarking.tool_metrics.evaluators.simple_evaluator import SimpleEvaluator
from src.benchmarking.tool_metrics.load_session import Loader
from src.benchmarking.tool_metrics.utils import QueryAndReference
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

    def run(self, dir_path: Path, session_file: Path | str, tools_description: Path | str) -> None:
        memory_logger = MemoryLogger()
        baseline_logger = BaselineLogger()

        base_recsum = RecsumDialogueSystem(embed_code=False, embed_tool=False)
        rag_recsum = RecsumDialogueSystem(embed_code=True, embed_tool=True)

        base_memory_bank = MemoryBankDialogueSystem(embed_code=False, embed_tool=False)
        rag_memory_bank = MemoryBankDialogueSystem(embed_code=True, embed_tool=True)

        full_baseline = DialogueBaseline("FullBaseline")
        last_baseline = DialogueBaseline("LastBaseline")

        algorithms_with_memory = [
            base_recsum,
            rag_recsum,
            base_memory_bank,
            rag_memory_bank
        ]

        baseline_algorithms = [
            full_baseline,
            last_baseline
        ]

        past_interactions = Loader.load_session(dir_path / session_file)
        tools = Loader.load_func_tools(dir_path / tools_description)

        query_and_reference = self.__execute_query_and_reference(past_interactions)
        query, reference = query_and_reference.query, query_and_reference.reference
        prompt = self.__prepare_query_for_the_first_stage(query)

        simple_evaluator = SimpleEvaluator()

        memory_metrics = Calculator.evaluate(
            algorithms_with_memory,
            [simple_evaluator],
            [past_interactions],
            prompt,
            reference,
            memory_logger,
            tools
        )

        baseline_metrics = Calculator.evaluate(
            baseline_algorithms,
            [simple_evaluator],
            [past_interactions],
            prompt,
            reference,
            baseline_logger,
            tools
        )

        print(memory_metrics)
        print(baseline_metrics)

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



if __name__ == "__main__":
    runner = Runner()
    runner.run(
        dir_path=Path("/Users/mikhailkharlamov/Documents/.../create-agents-md"),
        session_file="chat_20251022_001157_763_6010.messages.json",
        tools_description="chat_20251022_001157_763_6010.tools.json"
    )
