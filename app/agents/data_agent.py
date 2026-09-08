import json
from typing import Any

from app.core.llm import client, MODEL
from app.core.state import AgentState

from app.tools.data_tools import (
    inspect_dataset,
    analyze_target,
    compare_target_candidates,
)

from app.tools.ml_tools import (
    train_classifier,
    train_regressor,
)


DATA_TOOLS = [
    {
        "type": "function",
        "name": "inspect_dataset",
        "description": (
            "Perform a comprehensive inspection of a CSV dataset. "
            "Return dimensions, columns, data types, missing values, "
            "unique counts, duplicate rows, numeric statistics, "
            "categorical distributions, and sample rows."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV dataset.",
                }
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "compare_target_candidates",
        "description": (
            "Analyze dataset columns and identify possible target "
            "variables based on data type and cardinality."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV dataset.",
                }
            },
            "required": ["file_path"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "analyze_target",
        "description": (
            "Analyze a specific target column. Return its data type, "
            "unique values, class counts, class percentages, "
            "number of classes, and missing values."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV dataset.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Candidate target column.",
                },
            },
            "required": [
                "file_path",
                "target_column",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "train_classifier",
        "description": (
            "Train and compare Logistic Regression, Decision Tree, "
            "and Random Forest classifiers using preprocessing "
            "pipelines. Return metrics and a majority-class baseline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV dataset.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Classification target column.",
                },
            },
            "required": [
                "file_path",
                "target_column",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "train_regressor",
        "description": (
            "Train and evaluate a Random Forest regression model "
            "using a preprocessing pipeline."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the CSV dataset.",
                },
                "target_column": {
                    "type": "string",
                    "description": "Numeric regression target.",
                },
            },
            "required": [
                "file_path",
                "target_column",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


DATA_TOOL_FUNCTIONS = {
    "inspect_dataset": inspect_dataset,
    "compare_target_candidates": compare_target_candidates,
    "analyze_target": analyze_target,
    "train_classifier": train_classifier,
    "train_regressor": train_regressor,
}


def run_data_agent(
    state: AgentState,
    task: str,
    file_path: str | None = None,
) -> str:
    """
    Run the autonomous Data/ML specialist.

    The LLM decides which data/ML tool is appropriate,
    executes it, observes the result, and can request
    another tool when additional analysis is necessary.
    """

    if not file_path:
        result = (
            "No dataset file is available. "
            "A CSV file must be uploaded before data analysis "
            "or machine-learning tasks can be performed."
        )

        state.add_action(
            agent="DataAgent",
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
You are the Data and Machine Learning Specialist Agent
inside an autonomous task-execution system.

Your job is to independently determine what data or ML work
is required to accomplish the current task.

CURRENT TASK:
{task}

CONTEXT:
{json.dumps(context, indent=2, default=str)}

AVAILABLE TOOLS:
1. inspect_dataset
   - Comprehensive dataset profiling.

2. compare_target_candidates
   - Identify possible prediction targets.

3. analyze_target
   - Inspect a specific target and its class distribution.

4. train_classifier
   - Compare classification models.

5. train_regressor
   - Train a regression model.

AUTONOMOUS BEHAVIOR:
- Decide which tool is appropriate from the current task.
- You may call multiple tools.
- After observing a tool result, decide whether another tool
  is necessary.
- Do not assume that one tool call is always sufficient.
- Use previous results to avoid repeating unnecessary work.
- If the task asks for ML modeling, first understand the dataset
  and target before choosing the modeling approach.
- If a categorical target is identified, use classification.
- If a numeric continuous target is identified, regression may
  be appropriate.
- Never invent dataset values, metrics, or conclusions.
- Use the actual uploaded file path.
- Clearly explain limitations when the dataset is too small
  or the available evidence is insufficient.
- Once the task has been sufficiently completed, provide a
  concise evidence-based summary.

IMPORTANT:
The Manager Agent decides which specialist should work next.
You decide which DATA/ML TOOL is needed within your own work.
Do not delegate to other agents.
"""

    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=DATA_TOOLS,
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
                agent="DataAgent",
                action=task,
                result=result,
            )

            return result

        tool_outputs = []

        for call in tool_calls:

            arguments = json.loads(call.arguments)

            if "file_path" in arguments:
                arguments["file_path"] = file_path

            function = DATA_TOOL_FUNCTIONS.get(call.name)

            if function is None:
                result: Any = {
                    "error": (
                        f"Unknown DataAgent tool: {call.name}"
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
            tools=DATA_TOOLS,
            tool_choice="auto",
        )