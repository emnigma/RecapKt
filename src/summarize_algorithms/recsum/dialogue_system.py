import functools

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.summarize_algorithms.graph_nodes import (
    UpdateState,
    generate_response_node_recsum,
    should_continue_memory_update,
    update_memory_node,
)
from src.summarize_algorithms.models import RecsumDialogueState, Session, WorkflowNode
from src.summarize_algorithms.recsum.prompts import (
    MEMORY_UPDATE_PROMPT_TEMPLATE,
    RESPONSE_GENERATION_PROMPT_TEMPLATE,
)
from src.summarize_algorithms.recsum.summarizer import RecursiveSummarizer
from src.summarize_algorithms.response_generator import ResponseGenerator


class DialogueSystem:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.summarizer = RecursiveSummarizer(self.llm, MEMORY_UPDATE_PROMPT_TEMPLATE)
        self.response_generator = ResponseGenerator(
            self.llm, RESPONSE_GENERATION_PROMPT_TEMPLATE
        )
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(RecsumDialogueState)

        workflow.add_node(
            WorkflowNode.UPDATE_MEMORY.value,
            functools.partial(update_memory_node, self.summarizer),
        )
        workflow.add_node(
            WorkflowNode.GENERATE_RESPONSE.value,
            functools.partial(generate_response_node_recsum, self.response_generator),
        )

        workflow.set_entry_point(WorkflowNode.UPDATE_MEMORY.value)

        workflow.add_conditional_edges(
            WorkflowNode.UPDATE_MEMORY.value,
            should_continue_memory_update,
            {
                UpdateState.CONTINUE_UPDATE.value: WorkflowNode.UPDATE_MEMORY.value,
                UpdateState.FINISH_UPDATE.value: WorkflowNode.GENERATE_RESPONSE.value,
            },
        )

        workflow.add_edge(WorkflowNode.GENERATE_RESPONSE.value, END)

        return workflow.compile()

    def process_dialogue(
        self, sessions: list[Session], query: str
    ) -> RecsumDialogueState:
        initial_state = RecsumDialogueState(
            dialogue_sessions=sessions,
            current_session_index=0,
            query=query,
            response="",
        )

        return RecsumDialogueState(**self.graph.invoke(initial_state))
