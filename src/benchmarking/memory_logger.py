import json

from datetime import datetime
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.summarize_algorithms.core.models import DialogueState, MetricState, Session


class MemoryLogger(BaseLogger):
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState | None,
            metric: MetricState | None = None,
            is_return: bool = False
    ) -> None | dict[str, Any]:
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        if state is None:
            raise ValueError("'state' argument is necessary for MemoryLogger")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "response": getattr(state, "response", None),
            "memory": MemoryLogger._serialize_memories(state),
            "sessions": [s.to_dict() for s in sessions],
        }

        if metric is not None:
            record["metric_name"] = metric.metric.value
            record["metric_value"] = metric.value

        with open(self.log_dir / (system_name + str(iteration) + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.write("\n")

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

        if is_return:
            return record

    @staticmethod
    def _serialize_memories(state: DialogueState) -> dict[str, Any]:
        result = {}
        for name in ["text_memory_storage", "code_memory_storage", "tool_memory_storage"]:
            storage = getattr(state, name, None)
            if storage is not None:
                result[name] = storage.__dict__()

        text_memory = getattr(state, "text_memory", None)
        if text_memory is not None:
            result["text_memory"] = text_memory

        return result
