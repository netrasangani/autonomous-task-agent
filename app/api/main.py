from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.orchestrator import run_task


app = FastAPI(
    title="Autonomous Task Execution Agent",
    description=(
        "An autonomous AI agent that dynamically selects specialist "
        "agents and tools for data, ML, computer vision, and reporting."
    ),
    version="1.0.0",
)


UPLOADS_DIR = Path("uploads")
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


class TaskResponse(BaseModel):
    request_id: str
    goal: str
    file_path: str | None
    completed: bool
    iterations: int
    history: list
    results: list
    final_result: str


@app.get("/")
def root():
    return {
        "name": "Autonomous Task Execution Agent",
        "version": "1.0.0",
        "status": "running",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.post("/run", response_model=TaskResponse)
async def execute_task(
    goal: str = Form(...),
    file: UploadFile | None = File(None),
):
    """
    Execute an autonomous task.

    The user provides a high-level goal and may optionally upload
    a CSV dataset or image. The Manager Agent decides which
    specialist agent and tools are required.
    """

    goal = goal.strip()

    if not goal:
        raise HTTPException(
            status_code=400,
            detail="Goal cannot be empty.",
        )

    file_path = None

    if file is not None:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file has no filename.",
            )

        extension = Path(file.filename).suffix.lower()

        allowed_extensions = {
            ".csv",
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }

        if extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported file type. "
                    "Use CSV, PNG, JPG, JPEG, or WEBP."
                ),
            )

        safe_filename = f"{uuid4().hex}{extension}"
        destination = UPLOADS_DIR / safe_filename

        contents = await file.read()
        destination.write_bytes(contents)

        file_path = str(destination)

    try:
        result = run_task(
            goal=goal,
            file_path=file_path,
        )

        return TaskResponse(
            request_id=result["request_id"],
            goal=result["goal"],
            file_path=result["file_path"],
            completed=result["completed"],
            iterations=result["iterations"],
            history=result["history"],
            results=result["results"],
            final_result=result["final_result"],
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Task execution failed: {exc}",
        )