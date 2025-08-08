from typing import Optional, Type

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate

from src.summarize_algorithms.core.base_dialogue_system import BaseDialogueSystem
from src.summarize_algorithms.core.models import RecsumDialogueState, Session
from src.summarize_algorithms.recsum.prompts import (
    MEMORY_UPDATE_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
)
from src.summarize_algorithms.recsum.summarizer import RecursiveSummarizer


class RecsumDialogueSystem(BaseDialogueSystem):
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        super().__init__(llm)

    def _build_summarizer(self) -> RecursiveSummarizer:
        return RecursiveSummarizer(self.llm, MEMORY_UPDATE_PROMPT_TEMPLATE)

    def _get_response_prompt_template(self) -> PromptTemplate:
        return RESPONSE_GENERATION_PROMPT_TEMPLATE

    def _get_initial_state(
        self, sessions: list[Session], query: str
    ) -> RecsumDialogueState:
        return RecsumDialogueState(dialogue_sessions=sessions, query=query)

    @property
    def _get_dialogue_state_class(self) -> Type:
        return RecsumDialogueState
