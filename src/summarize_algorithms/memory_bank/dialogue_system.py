import functools

from typing import Optional

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.summarize_algorithms.graph_nodes import (
    UpdateState,
    generate_response_node_memory_bank,
    should_continue_memory_update,
    summarize_memory_node,
)
from src.summarize_algorithms.memory_bank.prompts import (
    RESPONSE_WITH_MEMORY_PROMPT,
    SESSION_SUMMARY_PROMPT,
)
from src.summarize_algorithms.memory_bank.summarizer import SessionSummarizer
from src.summarize_algorithms.models import (
    MemoryBankDialogueState,
    Session,
    WorkflowNode,
)
from src.summarize_algorithms.response_generator import ResponseGenerator


class DialogueSystem:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self.llm = llm or ChatOpenAI(model="gpt-4.1-mini", temperature=0.0)
        self.summarizer = SessionSummarizer(self.llm, SESSION_SUMMARY_PROMPT)
        self.response_generator = ResponseGenerator(
            self.llm, RESPONSE_WITH_MEMORY_PROMPT
        )
        self.graph = self._build_graph

    @property
    def _build_graph(self) -> CompiledStateGraph:
        workflow = StateGraph(MemoryBankDialogueState)

        workflow.add_node(
            WorkflowNode.UPDATE_MEMORY.value,
            functools.partial(summarize_memory_node, self.summarizer),
        )
        workflow.add_node(
            WorkflowNode.GENERATE_RESPONSE.value,
            functools.partial(
                generate_response_node_memory_bank, self.response_generator
            ),
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
    ) -> MemoryBankDialogueState:
        initial_state = MemoryBankDialogueState(
            dialogue_sessions=sessions,
            query=query,
        )

        return MemoryBankDialogueState(**self.graph.invoke(initial_state))
