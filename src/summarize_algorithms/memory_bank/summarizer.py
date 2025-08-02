from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field


class SessionMemory(BaseModel):
    summary_messages: list[str] = Field(description="Summary of session messages")


class SessionSummarizer:
    def __init__(self, llm: BaseChatModel, prompt_template: PromptTemplate) -> None:
        self.llm = llm
        self.prompt_template = prompt_template
        self.chain = self._build_chain()

    def _build_chain(self) -> RunnableSerializable[dict[str, Any], SessionMemory]:
        return cast(
            RunnableSerializable[dict, SessionMemory],
            self.prompt_template | self.llm.with_structured_output(SessionMemory),
        )

    def summarize(self, session_messages: str, session_id: int) -> list[str]:
        try:
            response = self.chain.invoke(
                {
                    "session_messages": session_messages,
                    "session_id": session_id,
                }
            )
            return response.summary_messages
        except Exception as e:
            raise ConnectionError(f"API request failed: {e}") from e
