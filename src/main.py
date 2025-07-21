from dialogue_system import DialogueSystem
from models import Message, Role, Session


def main() -> None:
    past_dialogue_sessions_data = [
        {
            "user": "Hi! I have a cat named Murka.",
            "bot": "Nice! I'm more of a dog person myself, but cats can be fun. What's Murka like?",
        },
        {
            "user": "Murka loves playing with laser pointers.",
            "bot": "Classic! Though I've heard it can frustrate them since they never catch anything. Speaking of\
             entertainment, do you read much?",
        },
        {
            "user": "I love reading fantasy books.",
            "bot": "Fantasy's hit or miss for me - I prefer sci-fi usually. Are you into any Italian authors? I've been\
             curious about their literature.",
        },
        {
            "user": "I'm planning to go to Italy.",
            "bot": "Perfect timing then! I found Rome overwhelming when I visited. You could read some Italian works to\
             prep for the trip.",
        },
        {
            "user": "I love cooking pasta.",
            "bot": "That'll be handy in Italy! I'm terrible at pasta - always turns out mushy. Maybe I should stick to\
             reading cookbooks instead of using them.",
        },
    ]

    current_query = "What would you recommend for me to read?"

    system = DialogueSystem()

    sessions = []
    for data in past_dialogue_sessions_data:
        session = Session(
            [Message(Role.USER, data["user"]), Message(Role.BOT, data["bot"])]
        )
        sessions.append(session)
    sessions.append(Session([Message(Role.USER, current_query)]))

    result = system.process_dialogue(sessions)

    print(f"Memory: {result['memory']}")
    print("-" * 50)
    print(f"Response: {result['response']}")


if __name__ == "__main__":
    main()