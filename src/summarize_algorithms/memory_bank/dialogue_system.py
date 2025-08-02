from typing import Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from src.summarize_algorithms.core.base_dialogue_system import BaseDialogueSystem
from src.summarize_algorithms.core.models import MemoryBankDialogueState, Session
from src.summarize_algorithms.memory_bank.prompts import (
    RESPONSE_WITH_MEMORY_PROMPT,
    SESSION_SUMMARY_PROMPT,
)
from src.summarize_algorithms.memory_bank.summarizer import SessionSummarizer


class DialogueSystem(BaseDialogueSystem):
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        super().__init__(llm)

    def _build_summarizer(self) -> SessionSummarizer:
        return SessionSummarizer(self.llm, SESSION_SUMMARY_PROMPT)

    def _get_response_prompt_template(self) -> PromptTemplate:
        return RESPONSE_WITH_MEMORY_PROMPT

    def _get_initial_state(
        self, sessions: list[Session], query: str
    ) -> MemoryBankDialogueState:
        return MemoryBankDialogueState(
            dialogue_sessions=sessions,
            query=query,
        )

    @property
    def _get_dialogue_state_class(self) -> Type:
        return MemoryBankDialogueState
