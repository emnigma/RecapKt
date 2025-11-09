from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import Runnable


class ResponseGenerator:
    def __init__(self,
                 llm: BaseChatModel,
                 prompt_template: PromptTemplate,
                 structure: dict[str, Any] | None = None,
                 tools: dict[str, Any] | None = None
                 ) -> None:
        self._llm = llm
        self._prompt_template = prompt_template
        self._structure = structure
        self._tools = tools
        self._chain = self._build_chain()

    def _build_chain(self) -> Runnable:
        if self._structure and not self._tools:
            structured_llm = self._llm.with_structured_output(self._structure)
            return self._prompt_template | structured_llm

        if self._tools and not self._structure:
            llm_with_tools = self._llm.bind_tools(self._tools)
            return self._prompt_template | llm_with_tools

        if self._tools and self._structure:
            tools = [self._get_return_action_plan(), *self._tools]
            llm_with_tools = self._llm.bind_tools(tools)
            return self._prompt_template | llm_with_tools

        return self._prompt_template | self._llm

    def _get_return_action_plan(self):
         return {
            "type": "function",
            "function": {
                "name": "return_action_plan",
                "description": "Return the final JSON action plan strictly matching the schema.",
                "parameters": self._structure,
            },
        }

    def generate_response(
        self, dialogue_memory: str, code_memory: str, tool_memory: str, query: str
    ) -> str:
        try:
            response = self._chain.invoke(
                {
                    "dialogue_memory": dialogue_memory,
                    "code_memory": code_memory,
                    "tool_memory": tool_memory,
                    "query": query,
                }
            )
            return response
        except Exception as e:
            raise ConnectionError(f"API request failed: {str(e)}") from e
