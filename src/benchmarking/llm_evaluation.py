from enum import Enum
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.benchmarking.prompts import (
    PAIRWISE_EVALUATION_MEMORY_PROMPT,
    PAIRWISE_EVALUATION_RESPONSE_PROMPT,
    SINGLE_EVALUATION_MEMORY_PROMPT,
    SINGLE_EVALUATION_RESPONSE_PROMPT,
)
from src.summarize_algorithms.core.models import OpenAIModels


class SingleResult(BaseModel):
    faithfulness_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Faithfulness"
    )
    informativeness_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Informativeness"
    )
    coherency_score: int = Field(
        ge=0, le=100, description="assessment by the criterion of Coherency"
    )


class ComparisonResult(Enum):
    OPTION_1_BETTER = "option 1 is better"
    OPTION_2_BETTER = "option 2 is better"
    DRAW = "draw"


class PairwiseResult(BaseModel):
    faithfulness: ComparisonResult = Field(
        description="Comparison of options based on Faithfulness"
    )
    informativeness: ComparisonResult = Field(
        description="Comparison of options based on Informativeness"
    )
    coherency: ComparisonResult = Field(
        description="Comparison of options based on Coherency"
    )


class LLMResponseEvaluation:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model=OpenAIModels.GPT_4_1.value, temperature=0.0)
        self.single_eval_prompt = SINGLE_EVALUATION_RESPONSE_PROMPT
        self.pairwise_eval_prompt = PAIRWISE_EVALUATION_RESPONSE_PROMPT
        self.single_eval_chain = self._build_single_eval_chain()
        self.pairwise_eval_chain = self._build_pairwise_eval_chain()

    def _build_single_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.single_eval_prompt | self.llm.with_structured_output(SingleResult)

    def _build_pairwise_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.pairwise_eval_prompt | self.llm.with_structured_output(
            PairwiseResult
        )

    def evaluate_single(self, context: str, memory: str, response: str) -> SingleResult:
        try:
            result = self.single_eval_chain.invoke(
                {"context": context, "memory": memory, "response": response}
            )
            return result
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e

    def evaluate_pairwise(
        self, context: str, memory: str, first_response: str, second_response: str
    ) -> PairwiseResult:
        try:
            result = self.pairwise_eval_chain.invoke(
                {
                    "context": context,
                    "memory": memory,
                    "first_response": first_response,
                    "second_response": second_response,
                }
            )
            return result
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e


class LLMMemoryEvaluation:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.single_eval_prompt = SINGLE_EVALUATION_MEMORY_PROMPT
        self.pairwise_eval_prompt = PAIRWISE_EVALUATION_MEMORY_PROMPT
        self.single_eval_chain = self._build_single_eval_chain()
        self.pairwise_eval_chain = self._build_pairwise_eval_chain()

    def _build_single_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.single_eval_prompt | self.llm.with_structured_output(SingleResult)

    def _build_pairwise_eval_chain(self) -> RunnableSerializable[dict[str, str], Any]:
        return self.pairwise_eval_prompt | self.llm.with_structured_output(
            PairwiseResult
        )

    def evaluate_single(self, ideal_memory: str, memory: str) -> SingleResult:
        try:
            result = self.single_eval_chain.invoke(
                {"generated_memory": memory, "ideal_memory": ideal_memory}
            )
            return result
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e

    def evaluate_pairwise(
        self, ideal_memory: str, first_memory: str, second_memory: str
    ) -> PairwiseResult:
        try:
            result = self.pairwise_eval_chain.invoke(
                {
                    "first_memory": first_memory,
                    "second_memory": second_memory,
                    "ideal_memory": ideal_memory,
                }
            )
            return result
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
