from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from src.summarize_algorithms.core.base_summarizer import BaseSummarizer


class RecursiveSummarizer(BaseSummarizer):
    def _build_chain(self) -> Runnable[dict[str, Any], str]:
        return self.prompt | self.llm | StrOutputParser()

    def summarize(self, previous_memory: str, dialogue_context: str) -> str:
        try:
            response = self.chain.invoke(
                {
                    "previous_memory": previous_memory,
                    "dialogue_context": dialogue_context,
                }
            )
            return response
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
