import functools

from typing import Any, Optional

from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from graph_nodes import (
    generate_response_node,
    should_continue_memory_update,
    update_memory_node,
)
from models import DialogueState, Session
from prompts import (
    MEMORY_UPDATE_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
)
from response_generator import ResponseGenerator
from summarizer import RecursiveSummarizer


class DialogueSystem:
    def __init__(self, llm: Optional[ChatOpenAI] = None) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
        self.summarizer = RecursiveSummarizer(self.llm, MEMORY_UPDATE_PROMPT_TEMPLATE)
        self.response_generator = ResponseGenerator(
            self.llm, RESPONSE_GENERATION_PROMPT_TEMPLATE
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(DialogueState)

        workflow.add_node(
            "update_memory", functools.partial(update_memory_node, self.summarizer)
        )
        workflow.add_node(
            "generate_response",
            functools.partial(generate_response_node, self.response_generator),
        )

        workflow.set_entry_point("update_memory")

        workflow.add_conditional_edges(
            "update_memory",
            should_continue_memory_update,
            {"continue_update": "update_memory", "finish_update": "generate_response"},
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def process_dialogue(
        self, sessions: list[Session], initial_memory: str = ""
    ) -> dict[str, Any]:
        initial_state = {
            "memory": initial_memory,
            "dialogue_sessions": sessions,
            "current_session_index": 0,
            "response": "",
        }

        return self.graph.invoke(initial_state)