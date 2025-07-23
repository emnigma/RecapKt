from src.recsum.models import DialogueState
from src.recsum.response_generator import ResponseGenerator
from src.recsum.summarizer import Summarizer


def update_memory_node(summarizer: Summarizer, state: DialogueState) -> DialogueState:
    if state.memory is None:
        state.memory = []

    if state.current_session_index < len(state.dialogue_sessions):
        current_dialogue_session = state.dialogue_sessions[state.current_session_index]

        if len(current_dialogue_session) > 0:
            new_memory = summarizer.summarize(
                state.latest_memory, str(current_dialogue_session)
            )
        else:
            new_memory = state.latest_memory

        state.memory.append(new_memory)
        state.current_session_index += 1

    return state


def generate_response_node(
    response_generator: ResponseGenerator, state: DialogueState
) -> DialogueState:

    final_response = response_generator.generate_response(
        state.latest_memory, state.current_context
    )

    state.response = final_response
    return state


def should_continue_memory_update(state: DialogueState) -> str:
    if state.current_session_index < len(state.dialogue_sessions):
        return "continue_update"
    else:
        return "finish_update"
