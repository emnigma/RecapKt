from langchain_core.prompts import PromptTemplate

BASELINE_PROMPT = PromptTemplate.from_template(
    """
You are an advanced AI language model capable of engaging in personality-based conversations.
Respond to the user based on the provided dialogue context. Craft a response that is natural and
conversational.
Dialog context: {context}
Query: {query}
The response to user is:"""
)

SINGLE_EVALUATION_RESPONSE_PROMPT = PromptTemplate.from_template(
    """
You are an impartial judge. You will be shown a Conversation Context, Personality of Speakers and Assistant Response.
#Faithfulness: Please evaluate whether the Assistant's response accurately represents the information provided in the
 conversation context and persona, without introducing false or misleading information.
#Informativeness: Please evaluate whether the Assistant's response provides useful, relevant, and comprehensive
 information that adds value to the conversation and addresses the user's needs effectively.
#Coherency: Please evaluate whether the Assistant's response maintains a coherent and logical flow of conversation
 based on the evolving context. A response with good context coherence can understand and respond appropriately to
 changes in conversation topics, providing smooth and sensible interactions.
Conversation Context: {context}
Personality: {memory}
Assistant Response: {response}
Begin your evaluation by providing a short explanation, then you must rate the Assistant Response
 on an integer score of 1 (very bad) to 100 (very good)."""
)

PAIRWISE_EVALUATION_RESPONSE_PROMPT = PromptTemplate.from_template(
    """
You are an expert evaluator tasked with comparing two AI assistant responses. You will evaluate them across three
 specific criteria and determine which response is better for each criterion.
## Input Information:
- Current Context: {context}
- Current Memory: {memory}
- Response 1: {first_response}
- Response 2: {second_response}

## Evaluation Criteria:

### 1. Faithfulness
Evaluate how well each response adheres to the given memory and maintains consistency with the established character
 traits. Consider:
- Does the response align with the assistant's defined personality?
- Are there any contradictions with previously established traits or information?
- Does the response maintain character consistency throughout?

### 2. Informativeness
Assess the quality and usefulness of information provided in each response. Consider:
- How comprehensive and relevant is the information?
- Does the response adequately address the user's query?
- Is the information accurate and helpful?
- Does it provide appropriate depth without being excessive?

### 3. Coherency
Evaluate the logical flow and contextual appropriateness of each response. Consider:
- Does the response flow logically and make sense in context?
- Are there smooth transitions between ideas?
- Does it appropriately respond to the conversation context?
- Is the response well-structured and easy to follow?

## Evaluation Options:
For each criterion, choose one of the following:
- Response 1 is better: Response 1 clearly outperforms Response 2
- Response 2 is better: Response 2 clearly outperforms Response 1
- Draw: Both responses are equally good

## Instructions:
Evaluate each criterion independently and provide your assessment for all three dimensions. Focus on objective
 evaluation based on the specific criteria rather than general preferences.
"""
)


SINGLE_EVALUATION_MEMORY_PROMPT = PromptTemplate.from_template("""
You are a meticulous and impartial evaluator. Your task is to assess the quality of the `Generated Memory`
 by comparing it to the `Ideal Memory`.

Your evaluation must be based on the following three criteria. You will compare the `Generated Memory` with the
 `Ideal Memory` (the "gold standard") to determine how well it aligns with it.

## Input Information:
- Generated Memory: {generated_memory}
- Ideal Memory: {ideal_memory}

### **Evaluation Criteria**

**1. Faithfulness (Accuracy)**
* **Question:** How accurately does the `Generated Memory` reproduce the facts and details presented in the
 `Ideal Memory`?
* **Assessment:** It should not contain information that contradicts or is absent from the `Ideal Memory`.
 Every piece of information in the `Generated Memory` should directly correspond to what is present in the
  `Ideal Memory`.

**2. Informativeness (Completeness and Relevance)**
* **Question:** Does the `Generated Memory` cover all the important information present in the `Ideal Memory`?
* **Assessment:** Evaluate whether it includes the most important topics, preferences, and facts outlined in the
 `Ideal Memory`. Does it omit critical details that are present in the `Ideal Memory`, or does it include trivial
  or irrelevant information?

**3. Coherency (Structure and Clarity)**
* **Question:** Is the `Generated Memory` well-organized, logical, and easy to understand?
* **Assessment:** The information should be structured in a clear and comprehensible way.
 Compare its structure and clarity to the `Ideal Memory`. Poor organization, confusing wording, or chaotic
  presentation should lower the score.

### **Score and Output Format**

Start your evaluation with a brief but thorough **Justification**. Justify your score by pointing out specific examples,
 strengths, or weaknesses in the `Generated Memory` compared to the `Ideal Memory`.

Then, assign an integer **Score** from 1 to 100 using the following scale:

* **81–100 (Very Good):** The `Generated Memory` is nearly identical to the `Ideal Memory` in terms of content,
 structure, and detail. It is accurate, comprehensive, and clear.
* **61–80 (Good):** The `Generated Memory` is mostly accurate and captures the main points, but may have minor omissions
 or slight structural issues compared to the `Ideal Memory`.
* **41–60 (Fair):** The `Generated Memory` reflects some key information but misses important details or contains minor
 inaccuracies compared to the `Ideal Memory`. Structure may be confusing.
* **21–40 (Poor):** The `Generated Memory` contains significant omissions, inaccuracies, or is poorly structured and
 difficult to understand.
* **1–20 (Very Poor):** The `Generated Memory` is largely inaccurate, includes contradictory information, or fails to
 convey the essence of the `Ideal Memory`.
""")

PAIRWISE_EVALUATION_MEMORY_PROMPT = PromptTemplate.from_template(
    """
You are an expert evaluator comparing two AI-generated memories against an ideal (gold standard) memory. Evaluate across
 three criteria and decide which memory is better for each criterion.

## Input:
- Ideal Memory (Gold Standard): {ideal_memory}
- Memory 1: {first_memory}
- Memory 2: {second_memory}

## Criteria:
1. Faithfulness
   - Consistency with the gold memory: accuracy of facts and details.
   - Adherence to the assistant's persona and prior knowledge.
2. Informativeness
   - Completeness: coverage of key information from the ideal memory.
   - Clarity and usefulness: relevance without unnecessary details.
3. Coherency
   - Logical structure and flow.
   - Readability and smooth transitions.

For each criterion, output one of:
- Memory 1 is better
- Memory 2 is better
- Draw: Both are equally good

Provide a brief rationale (1–2 sentences) only if the distinction is subtle or non-obvious.
    """
)
