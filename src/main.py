from src.summarize_algorithms.core.models import BaseBlock, Session
from src.summarize_algorithms.memory_bank.dialogue_system import (
    MemoryBankDialogueSystem,
)


def main() -> None:
    role1 = "user"
    role2 = "assistant"
    past_dialogue_sessions_data = [
        {
            role1: "Hi! I have a cat named Murka.",
            role2: "Nice! I'm more of a dog person myself, but cats can be fun. What's Murka like?",
        },
        {
            role1: "Murka loves playing with laser pointers.",
            role2: "Classic! Though I've heard it can frustrate them since they never catch anything. "
            "Speaking of entertainment, do you read much?",
        },
        {
            role1: "I love reading fantasy books.",
            role2: "Fantasy's hit or miss for me - I prefer sci-fi usually. Are you into any Italian authors? "
            "I've been curious about their literature.",
        },
        {
            role1: "I'm planning to go to Italy.",
            role2: "Perfect timing then! I found Rome overwhelming when I visited. "
            "You could read some Italian works to prep for the trip.",
        },
        {
            role1: "I love cooking pasta.",
            role2: "That'll be handy in Italy! I'm terrible at pasta - always turns out mushy. "
            "Maybe I should stick to reading cookbooks instead of using them.",
        },
    ]

    current_query = "What would you recommend for me to read?"

    system = MemoryBankDialogueSystem()

    sessions = []
    for data in past_dialogue_sessions_data:
        session = Session(
            [BaseBlock(role1, data[role1]), BaseBlock(role2, data[role2])]
        )
        sessions.append(session)

    result = system.process_dialogue(sessions, current_query)

    print(f"Response: {result.response}")


if __name__ == "__main__":
    main()
