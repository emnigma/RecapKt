from src.summarize_algorithms.core.base_summarizer import BaseSummarizer
from src.summarize_algorithms.core.models import (
    DialogueState,
    MemoryBankDialogueState,
    RecsumDialogueState,
    UpdateState,
)
from src.summarize_algorithms.core.response_generator import ResponseGenerator


def update_memory_node(
    summarizer_instance: BaseSummarizer, state: DialogueState
) -> DialogueState:
    current_dialogue_session = state.dialogue_sessions[state.current_session_index]

    if isinstance(state, RecsumDialogueState):
        new_memory = summarizer_instance.summarize(
            state.latest_memory, str(current_dialogue_session)
        )
        state.memory.append(new_memory)
        state.current_session_index += 1
        return state
    elif isinstance(state, MemoryBankDialogueState):
        new_memory = summarizer_instance.summarize(
            str(current_dialogue_session), state.current_session_index
        )
        state.memory_storage.add_memory(new_memory, state.current_session_index)
        state.current_session_index += 1
        return state
    else:
        raise TypeError(
            f"Unsupported status type for update_memory_node: {type(state)}"
        )


def generate_response_node(
    response_generator_instance: ResponseGenerator, state: DialogueState
) -> DialogueState:
    if isinstance(state, RecsumDialogueState):
        final_response = response_generator_instance.generate_response(
            state.latest_memory, str(state.dialogue_sessions[-1]), state.query
        )
    elif isinstance(state, MemoryBankDialogueState):
        top_relevant_memory = state.memory_storage.find_similar(
            state.query, state.current_session_index
        )
        final_response = response_generator_instance.generate_response(
            "\n".join(top_relevant_memory),
            str(state.dialogue_sessions[-1]),
            state.query,
        )
    else:
        raise TypeError(
            f"Unsupported status type for update_memory_node: {type(state)}"
        )

    state.response = final_response
    return state


def should_continue_memory_update(state: DialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions):
        return UpdateState.CONTINUE_UPDATE.value
    else:
        return UpdateState.FINISH_UPDATE.value
