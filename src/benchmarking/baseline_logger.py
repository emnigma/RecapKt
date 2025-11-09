import json

from datetime import datetime
from typing import Any

from src.benchmarking.base_logger import BaseLogger
from src.summarize_algorithms.core.models import DialogueState, MetricState, Session


class BaselineLogger(BaseLogger):
    def log_iteration(
            self,
            system_name: str,
            query: str,
            iteration: int,
            sessions: list[Session],
            state: DialogueState | None = None,
            metrics: list[MetricState] | None = None
    ) -> dict[str, Any]:
        self.logger.info(f"Logging iteration {iteration} to {self.log_dir}")

        record = {
            "timestamp": datetime.now().isoformat(),
            "iteration": iteration,
            "system": system_name,
            "query": query,
            "sessions": [s.to_dict() for s in sessions],
        }

        if metrics is not None:
            metrics_dict = [
                {"metric_name": metric.metric.value, "metric_value": metric.value}
                for metric in metrics
            ]
            record["metric"] = metrics_dict

        with open(self.log_dir / (system_name + str(iteration) + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, indent=4))
            f.write("\n")

        self.logger.info(f"Saved successfully iteration {iteration} to {self.log_dir}")

        return record
