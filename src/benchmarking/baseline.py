import os

from typing import Any, List, Optional

from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from src.benchmarking.baseline_logger import BaselineLogger
from src.benchmarking.prompts import BASELINE_PROMPT
from src.summarize_algorithms.core.dialog import Dialog
from src.summarize_algorithms.core.models import DialogueState, OpenAIModels, Session


class DialogueBaseline(Dialog):
    def __init__(self, system_name: str, llm: Optional[BaseChatModel] = None) -> None:
        load_dotenv()

        self.system_name = system_name

        api_key: str | None = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            self.llm = llm or ChatOpenAI(
                model=OpenAIModels.GPT_5_MINI.value,
                api_key=SecretStr(api_key)
            )
        else:
            raise ValueError("OPENAI_API_KEY environment variable is not loaded")

        self.prompt_template = BASELINE_PROMPT
        self.chain = self._build_chain()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0.0

        self.baseline_logger = BaselineLogger()

    def _build_chain(
            self,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None
    ) -> Runnable:
        if structure and not tools:
            structured_llm = self.llm.with_structured_output(structure)
            return self.prompt_template | structured_llm

        if tools and not structure:
            llm_with_tools = self.llm.bind_tools(tools)
            return self.prompt_template | llm_with_tools

        if tools and structure:
            tools = [DialogueBaseline._get_return_action_plan(structure), *tools]
            llm_with_tools = self.llm.bind_tools(tools)
            return self.prompt_template | llm_with_tools

        return self.prompt_template | self.llm | StrOutputParser()

    @staticmethod
    def _get_return_action_plan(structure):
        return {
            "type": "function",
            "function": {
                "name": "return_action_plan",
                "description": "Return the final JSON action plan strictly matching the schema.",
                "parameters": structure,
            },
        }

    #TODO remove iteration argument and logging
    def process_dialogue(
            self,
            sessions: List[Session],
            query: str,
            structure: dict[str, Any] | None = None,
            tools: list[dict[str, Any]] | None = None,
            iteration: int | None = None
    ) -> DialogueState:
        if structure is not None or tools is not None:
            chain = self._build_chain(structure, tools)
        else:
            chain = self.chain

        context_messages = []
        for session in sessions:
            for message in session.messages:
                context_messages.append(f"{message.role}: {message.content}")
        context = "\n".join(context_messages)
        with get_openai_callback() as cb:
            result = chain.invoke({"context": context, "query": query})

            self.prompt_tokens += cb.prompt_tokens
            self.completion_tokens += cb.completion_tokens
            self.total_cost += cb.total_cost

        if iteration is not None:
            self.baseline_logger.log_iteration(
                system_name=self.system_name,
                query=query,
                iteration=iteration,
                sessions=sessions
            )

        return DialogueState(
            dialogue_sessions=sessions,
            query=query,
            _response=result,
            code_memory_storage=None,
            tool_memory_storage=None
        )
