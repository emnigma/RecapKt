import logging

from src.benchmarking.baseline import DialogueBaseline
from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.memory_logger import MemoryLogger
from src.benchmarking.tool_metrics.calculator import Calculator
from src.benchmarking.tool_metrics.evaluators.simple_evaluator import SimpleEvaluator
from src.benchmarking.tool_metrics.load_session import SessionLoader
from src.summarize_algorithms.core.models import BaseBlock
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)
from src.summarize_algorithms.recsum.dialogue_system import RecsumDialogueSystem


class Runner:
    def __init__(self):
        self.logger = logging.getLogger()

    def run(self):
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

        past_interactions = SessionLoader.load_session("chat_20251022_001157_763_6010.messages.json")

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

        simple_evaluator = SimpleEvaluator()

        memory_metrics = Calculator.evaluate(
            algorithms_with_memory,
            [simple_evaluator],
            [past_interactions],
            query,
            reference,
            memory_logger
        )

        baseline_metrics = Calculator.evaluate(
            baseline_algorithms,
            [simple_evaluator],
            [past_interactions],
            query,
            reference,
            baseline_logger
        )

        print(memory_metrics)
        print(baseline_metrics)


if __name__ == "__main__":
    runner = Runner()
    runner.run()
