from langchain_core.prompts import PromptTemplate

PLAN_PROMPT = PromptTemplate.from_template(
    """
    **Instruction: Return a Strict JSON Action Plan**
    
    **You are given:**
        •	A multi-session conversation between a user and an LLM, including prior tool calls and results.
        •	A catalog of available tools/functions with their argument specifications.
        •	The user’s latest request that must be addressed now.
    Your job is to plan how to answer the latest user request without executing any tools. Instead, produce a strict JSON action plan that lists, in correct order, every tool you would call (with arguments) and any non-tool actions (reasoning, synthesis, checking constraints) required to deliver the final answer.
    Do not invent tools. If some data is unknown at planning time, set argument values to null and add an assumption that states what would be discovered by executing that step.
    Return only JSON that conforms to the schema below. Do not include any commentary, Markdown, or backticks. Do not add fields not defined in the schema. Do not include trailing commas.
    **Inputs Provided**
        •	ConversationHistory: the full or summarized transcript of sessions 1..n.
        •	MemoryArtifacts: distilled notes/summaries retrieved by the memory algorithm.
        •	ToolsCatalog: the tools you may plan to call (names, descriptions, and argument specs).
        •	LatestUserRequest: the current user message to address in session n+1.
    **Planning Guidelines**
        •	Order steps exactly as they should be executed.
        •	Use depends_on to express explicit dependencies on previous steps.
        •	Use kind = “tool_call” for planned tool invocations and provide args.
        •	Use kind = “action” for reasoning, analysis, or transformations not covered by tools.
        •	Use kind = “final_answer” for the final composition plan only; do not write the answer itself.
        •	If an argument value is unknown, put null and add an entry in assumptions describing what would be obtained and how it affects subsequent steps.
        •	Reference memory you rely on in memory_used so we can validate memory utilization.
        •	Keep descriptions concise and operational. Avoid speculative content beyond stated assumptions.
    **Tools Catalog**
        •	You may only plan calls to tools declared in ToolsCatalog.
        •	Use the exact tool name in plan_stepsi.name when kind = “tool_call”.
        •	Populate args according to the tool’s argument spec. Unknown values must be null and justified in assumptions.
    **Validation Rules**
        •	Output must be valid JSON per the schema above.
        •	Steps must be listed in intended execution order.
        •	depends_on must reference only prior step ids.
        •	kind field governs which optional fields appear:
        •	tool_call: include args and expected; omit answer_template and collect_from.
        •	action: include expected; omit args, answer_template, collect_from.
        •	final_answer: include answer_template and collect_from; omit args.
        •	Do not execute tools. Do not include tool outputs, only expected.
    **Example Output**
    {
      "version": "1.0",
      "user_request": "Find the cheapest flight from NYC to SFO this weekend and summarize baggage rules.",
      "context_summary": "User often prefers red-eye flights; prior session stored preferred airlines: Delta, Alaska.",
      "memory_used": [
        { "source": "memory_bank", "key": "preferences.airlines", "excerpt": "Delta, Alaska" },
        { "source": "session", "key": "s7.note", "excerpt": "Red-eye flights acceptable; price-sensitive." }
      ],
      "assumptions": [
        "If multiple flights tie on price, choose the earliest arrival.",
        "Baggage policy will be retrieved per carrier after selecting candidate flights."
      ],
      "risks": [
        "Weekend surge pricing may change between planning and execution."
      ],
      "plan_steps": [
        {
          "id": "s1",
          "kind": "action",
          "name": "narrow_search_criteria",
          "description": "Set date range to upcoming weekend, departure NYC airports, arrival SFO; prefer Delta/Alaska and red-eyes.",
          "depends_on": [],
          "expected": "Concrete filters for search tools."
        },
        {
          "id": "s2",
          "kind": "tool_call",
          "name": "search_flights",
          "description": "Search flights with filters from s1.",
          "depends_on": ["s1"],
          "args": {
            "origin": "NYC",
            "destination": "SFO",
            "start_date": null,
            "end_date": null,
            "max_stops": 1,
            "airlines": ["Delta", "Alaska"]
          },
          "expected": "List of flights with prices and fare classes."
        },
        {
          "id": "s3",
          "kind": "action",
          "name": "select_cheapest",
          "description": "From s2 results, pick cheapest flight; break ties by earliest arrival.",
          "depends_on": ["s2"],
          "expected": "Chosen flight id and carrier."
        },
        {
          "id": "s4",
          "kind": "tool_call",
          "name": "get_baggage_policy",
          "description": "Retrieve baggage rules for chosen carrier and fare.",
          "depends_on": ["s3"],
          "args": {
            "carrier_code": null,
            "fare_class": null
          },
          "expected": "Baggage allowance and fees."
        },
        {
          "id": "s5",
          "kind": "final_answer",
          "name": "compose_answer",
          "description": "Draft final response with selected flight and baggage summary.",
          "depends_on": ["s3", "s4"],
          "answer_template": "Flight: {{flight_summary_from_s3}}. Baggage: {{baggage_points_from_s4}}. Note surge pricing.",
          "collect_from": ["s3", "s4"]
        }
      ]
    }
    **Return Format**
        •	Return only the JSON object. No Markdown, no preamble, no postscript.
    
    **Here is a user request, in response to which you need to make an action plan:**
    {query}
    """
)
