import json
from typing import Any

from app.core.llm import client, MODEL
from app.core.state import AgentState
from app.tools.vision_tools import inspect_image


VISION_TOOLS = [
    {
        "type": "function",
        "name": "inspect_image",
        "description": (
            "Inspect an uploaded image and return its format, "
            "dimensions, color mode, channels, aspect ratio, "
            "and file size."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the image file.",
                }
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
        "strict": True,
    }
]


VISION_TOOL_FUNCTIONS = {
    "inspect_image": inspect_image,
}


def run_vision_agent(
    state: AgentState,
    task: str,
    file_path: str | None = None,
) -> str:
    """
    Run the autonomous Computer Vision specialist.

    The agent decides when image inspection is required,
    executes the available vision tool, observes the result,
    and produces an evidence-based response.
    """

    if not file_path:
        result = (
            "No image file is available. "
            "An image must be uploaded before computer-vision "
            "analysis can be performed."
        )

        state.add_action(
            agent="VisionAgent",
            action=task,
            result=result,
        )

        return result

    context = {
        "goal": state.goal,
        "current_task": task,
        "file_path": file_path,
        "execution_history": state.history,
        "previous_results": state.results,
    }

    prompt = f"""
You are the Computer Vision Specialist Agent inside an
autonomous task-execution system.

Your responsibility is to perform image-related work required
by the Manager Agent.

CURRENT TASK:
{task}

CONTEXT:
{json.dumps(context, indent=2, default=str)}

AVAILABLE TOOL:
- inspect_image

AUTONOMOUS BEHAVIOR:
- Decide whether image inspection is required.
- Use the uploaded image path.
- Execute the appropriate tool.
- Carefully inspect the returned result.
- Do not invent image properties or observations.
- If the available tool does not provide enough information
  to answer a requested task, clearly state the limitation.
- Provide a concise factual summary after completing the task.
- Use previous results when relevant and avoid unnecessary
  repeated tool calls.
- Do not delegate to other agents.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=VISION_TOOLS,
        tool_choice="auto",
    )

    while True:

        tool_calls = [
            item
            for item in response.output
            if item.type == "function_call"
        ]

        if not tool_calls:
            result = response.output_text

            state.add_action(
                agent="VisionAgent",
                action=task,
                result=result,
            )

            return result

        tool_outputs = []

        for call in tool_calls:

            arguments = json.loads(call.arguments)

            if "file_path" in arguments:
                arguments["file_path"] = file_path

            function = VISION_TOOL_FUNCTIONS.get(call.name)

            if function is None:
                result: Any = {
                    "error": (
                        f"Unknown VisionAgent tool: {call.name}"
                    )
                }

            else:
                try:
                    result = function(**arguments)

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

        response = client.responses.create(
            model=MODEL,
            input=tool_outputs,
            previous_response_id=response.id,
            tools=VISION_TOOLS,
            tool_choice="auto",
        )