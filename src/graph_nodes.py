from models import DialogueState
from response_generator import ResponseGenerator
from summarizer import Summarizer


def update_memory_node(summarizer: Summarizer, state: DialogueState) -> DialogueState:
    current_session_index = state["current_session_index"]
    dialogue_sessions = state["dialogue_sessions"]
    previous_memory = state["memory"]

    if current_session_index < len(dialogue_sessions):
        current_dialogue_session = dialogue_sessions[current_session_index]

        if len(current_dialogue_session) > 0:
            new_memory = summarizer.summarize(
                previous_memory, str(current_dialogue_session)
            )
        else:
            new_memory = previous_memory

        return {
            "dialogue_sessions": dialogue_sessions,
            "current_session_index": current_session_index + 1,
            "memory": new_memory,
            "response": state["response"],
        }
    else:
        return state


def generate_response_node(
    response_generator: ResponseGenerator, state: DialogueState
) -> DialogueState:
    latest_memory = state["memory"]

    if state["dialogue_sessions"]:
        current_dialogue_context = str(state["dialogue_sessions"][-1])
    else:
        current_dialogue_context = ""

    final_response = response_generator.generate_response(
        latest_memory, current_dialogue_context
    )

    return {
        "dialogue_sessions": state["dialogue_sessions"],
        "current_session_index": state["current_session_index"],
        "memory": state["memory"],
        "response": final_response,
    }


def should_continue_memory_update(state: DialogueState) -> str:
    if state["current_session_index"] < len(state["dialogue_sessions"]):
        return "continue_update"
    else:
        return "finish_update"
