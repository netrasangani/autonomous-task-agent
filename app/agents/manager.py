import json

from app.core.llm import client, MODEL
from app.core.state import AgentState

from app.agents.data_agent import run_data_agent
from app.agents.vision_agent import run_vision_agent
from app.agents.report_agent import run_report_agent


MANAGER_TOOLS = [
    {
        "type": "function",
        "name": "delegate_to_data_agent",
        "description": (
            "Delegate data analysis or machine learning work to "
            "DataAgent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Specific data or ML task."
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delegate_to_vision_agent",
        "description": (
            "Delegate image analysis or computer vision work "
            "to VisionAgent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Specific computer vision task."
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "delegate_to_report_agent",
        "description": (
            "Delegate final report generation to ReportAgent "
            "after enough evidence has been collected."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Specific reporting task."
                }
            },
            "required": ["task"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def manager_decide(
    state: AgentState,
    file_path: str | None = None,
) -> str:

    context = {
        "goal": state.goal,
        "file_path": file_path,
        "history": state.history,
        "results": state.results,
    }

    prompt = f"""
You are the Manager Agent of an autonomous AI task-solving system.

Your job is to decide what should happen NEXT to accomplish
the user's goal.

USER GOAL:
{state.goal}

AVAILABLE FILE:
{file_path}

CURRENT STATE:
{json.dumps(context, indent=2, default=str)}

AVAILABLE SPECIALISTS:

DataAgent:
- CSV/data analysis
- statistics
- target identification
- preprocessing
- machine learning
- model evaluation

VisionAgent:
- image analysis
- computer vision

ReportAgent:
- final synthesis
- conclusions
- recommendations
- professional PDF report

RULES:

1. Do NOT follow a predefined workflow.
2. Inspect the current goal and previous results.
3. Decide the single most useful next action.
4. Delegate that action to the appropriate specialist.
5. After a specialist returns a result, reassess the situation.
6. A specialist can be called again when additional work is needed.
7. Do not call ReportAgent until sufficient evidence exists.
8. If a report was requested, call ReportAgent once the analysis
   is sufficiently complete.
9. After ReportAgent successfully generates the report, finish.
10. Never invent results.
11. If required input is missing, have the relevant specialist
    explain what is required.
12. You are the decision-maker. Do not perform specialist work
    yourself.

Use your function tools to delegate work.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=MANAGER_TOOLS,
        tool_choice="auto",
    )

    while True:

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            state.completed = True
            return response.output_text

        tool_outputs = []

        for call in tool_calls:

            arguments = json.loads(call.arguments)

            try:

                if call.name == "delegate_to_data_agent":

                    result = run_data_agent(
                        state=state,
                        task=arguments["task"],
                        file_path=file_path,
                    )

                elif call.name == "delegate_to_vision_agent":

                    result = run_vision_agent(
                        state=state,
                        task=arguments["task"],
                        file_path=file_path,
                    )

                elif call.name == "delegate_to_report_agent":

                    result = run_report_agent(
                        state=state,
                        task=arguments["task"],
                    )

                    state.completed = True

                else:

                    result = {
                        "error": f"Unknown tool: {call.name}"
                    }

            except Exception as exc:

                result = {
                    "error": str(exc)
                }

            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(
                        result,
                        default=str,
                    ),
                }
            )

        if state.completed:
            return "Task completed successfully."

        response = client.responses.create(
            model=MODEL,
            input=tool_outputs,
            previous_response_id=response.id,
            tools=MANAGER_TOOLS,
            tool_choice="auto",
        )