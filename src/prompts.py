from langchain_core.prompts import PromptTemplate

MEMORY_UPDATE_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are an advanced AI language model with the ability to store and update a memory to keep track of
key personality information for both the user and the bot. You will receive a previous memory and
dialogue context. Your goal is to update the memory by incorporating the new personality information.
To successfully update the memory, follow these steps:
1.Carefully analyze the existing memory and extract the key personality of the user and bot from it.
2. Consider the dialogue context provided to identify any new or changed personality that needs to
be incorporated into the memory.
3. Combine the old and new personality information to create an updated representation of the user and
bot’s traits.
4. Structure the updated memory in a clear and concise manner, ensuring it does not exceed 20 sentences.
Remember, the memory should serve as a reference point to
maintain continuity in the dialogue and help you respond accurately to the user based on their personality.
Previous Memory: {previous_memory}
Session Context: {dialogue_context}"""
)

RESPONSE_GENERATION_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You will be provided with a memory containing personality information for both yourself and the user.
Your goal is to respond accurately to the user based on the personality traits and dialogue context.
Follow these steps to successfully complete the task:
1. Analyze the provided memory to extract the key personality traits for both yourself and the user.
2. Review the dialogue history to understand the context and flow of the conversation.
3. Utilize the extracted personality traits and dialogue context to formulate an appropriate response.
4. If no specific personality trait is applicable, respond naturally as a human would.
5. Pay attention to the relevance and importance of the personality information, focusing on capturing
the most significant aspects while maintaining the overall coherence of the memory.
Previous Memory: {latest_memory}
Current Context: {current_dialogue_context}"""
)