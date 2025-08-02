from enum import Enum

from src.summarize_algorithms.memory_bank.summarizer import SessionSummarizer
from src.summarize_algorithms.models import MemoryBankDialogueState, RecsumDialogueState
from src.summarize_algorithms.recsum.summarizer import Summarizer
from src.summarize_algorithms.response_generator import ResponseGenerator


class UpdateState(Enum):
    CONTINUE_UPDATE = "continue_update"
    FINISH_UPDATE = "finish_update"


def update_memory_node(
    summarizer: Summarizer, state: RecsumDialogueState
) -> RecsumDialogueState:
    current_dialogue_session = state.dialogue_sessions[state.current_session_index]

    new_memory = summarizer.summarize(
        state.latest_memory, str(current_dialogue_session)
    )

    state.memory.append(new_memory)
    state.current_session_index += 1

    return state


def generate_response_node_recsum(
    response_generator: ResponseGenerator, state: RecsumDialogueState
) -> RecsumDialogueState:
    final_response = response_generator.generate_response(
        state.latest_memory, str(state.dialogue_sessions[-1]), state.query
    )

    state.response = final_response
    return state


def summarize_memory_node(
    summarizer: SessionSummarizer, state: MemoryBankDialogueState
) -> MemoryBankDialogueState:
    current_dialogue_session = state.dialogue_sessions[state.current_session_index]

    new_memory = summarizer.summarize(
        str(current_dialogue_session), state.current_session_index
    )
    state.memory_storage.add_memory(new_memory, state.current_session_index)

    state.current_session_index += 1

    return state


def generate_response_node_memory_bank(
    response_generator: ResponseGenerator, state: MemoryBankDialogueState
) -> MemoryBankDialogueState:
    top_relevant_memory = state.memory_storage.find_similar(
        state.query, state.current_session_index
    )
    final_response = response_generator.generate_response(
        "\n".join(top_relevant_memory), str(state.dialogue_sessions[-1]), state.query
    )

    state.response = final_response
    return state


def should_continue_memory_update(state: RecsumDialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions) - 1:
        return UpdateState.CONTINUE_UPDATE.value
    else:
        return UpdateState.FINISH_UPDATE.value
