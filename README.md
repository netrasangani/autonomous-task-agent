# 🤖 Autonomous Task Execution Agent

> An AI-powered autonomous agent that analyzes datasets, performs machine learning tasks, inspects images, and generates professional PDF reports through a multi-agent workflow.

---

## 🚀 Overview

The **Autonomous Task Execution Agent** is designed to handle complex tasks with minimal manual intervention.

Instead of requiring the user to specify every individual operation, the system uses a **Manager Agent** to determine which specialized agent and tools are required to complete the task.

### ✨ What the system can do

- 📊 Analyze and inspect datasets
- 🎯 Identify potential prediction targets
- 🤖 Determine suitable machine learning approaches
- 🧠 Train and evaluate ML models
- 🖼️ Inspect and analyze images
- 📝 Interpret analysis results
- 📄 Generate professional PDF reports
- 🔄 Coordinate multiple specialized agents
- 🆔 Track each request using a unique Request ID
- 📋 Maintain execution history and results

---

## 🏗️ System Architecture

```text
                         👤 User
                           │
                           ▼
                  ┌─────────────────┐
                  │  Manager Agent  │
                  └────────┬────────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
      ┌────────────┐ ┌────────────┐ ┌────────────┐
      │ Data Agent │ │Vision Agent│ │Report Agent│
      └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
            │              │              │
            ▼              ▼              ▼
       📊 Data & ML     🖼️ Image       📄 PDF
          Tools          Tools        Generation
            │              │              │
            └──────────────┼──────────────┘
                           │
                           ▼
                   ✅ Final Result

```

---
# How to Run

## 1. Prerequisites

Before running the project, make sure the following are installed:

* Python 3.10+
* Git
* pip
* OpenAI API key

Verify the installations:

```bash
python --version
git --version
pip --version
```

---

## 2. Clone the Repository

Clone the repository from GitHub:

```bash
git clone https://github.com/netrasangani/autonomous-task-agent.git
```

Navigate into the project directory:

```bash
cd autonomous-task-agent
```

---

## 3. Create a Virtual Environment

Create a virtual environment named `.venv`:

```bash
python -m venv .venv
```

---

## 4. Activate the Virtual Environment

### macOS / Linux

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

After activation, `(.venv)` should appear in the terminal.

---

## 5. Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Configure the Environment Variables

Create a `.env` file in the root directory of the project.

Add your OpenAI API key:

```env
OPENAI_API_KEY=your_openai_api_key
```

The `.env` file should **not** be committed to GitHub.

---

# Running the Application

The application consists of two components:

* **FastAPI Backend**
* **Streamlit Frontend**

Both need to run at the same time.

You will therefore need **two terminals**.

---

## 7. Start the Backend

Open **Terminal 1**.

Navigate to the project directory:

```bash
cd autonomous-task-agent
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Start the FastAPI server:

```bash
uvicorn app.api.main:app --reload
```

The backend will be available at:

**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

FastAPI's interactive documentation will be available at:

**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

Keep this terminal running.

---

## 8. Start the Frontend

Open **Terminal 2**.

Navigate to the project directory:

```bash
cd autonomous-task-agent
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Start the Streamlit frontend:

```bash
streamlit run ui.py
```

The frontend will be available at:

**[http://localhost:8501](http://localhost:8501)**

Open this URL in your browser.

Keep this terminal running.

---

# Using the Application

## 9. Enter a Task

Once the Streamlit interface is open, enter a natural-language task describing what you want the agent to accomplish.

For example:

> Analyze the uploaded dataset and autonomously determine the most appropriate machine learning approach. Explore the data, identify the likely prediction target and preprocessing requirements, train and evaluate a suitable model, interpret the results, and generate a professional PDF report with findings, conclusions, and recommendations.

---

## 10. Upload Input

Upload the required input file through the Streamlit interface.

The application supports:

* CSV datasets
* Images

For a machine-learning task, upload the CSV dataset you want the agent to analyze.

For an image-analysis task, upload the image you want the agent to inspect.

---

## 11. Start the Task

Submit the task from the frontend.

The request follows this flow:

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
Manager Agent
  ↓
Specialized Agent
  ↓
Results
  ↓
Streamlit Frontend
```

The Manager Agent determines which specialized agent should handle the task.

---

## 12. Data Agent

For dataset and machine-learning tasks, the Manager Agent delegates the work to the Data Agent.

The Data Agent can:

1. Inspect the dataset.
2. Compare possible target columns.
3. Analyze the target.
4. Determine the appropriate ML approach.
5. Train a classifier.
6. Train a regressor.
7. Evaluate the model.
8. Return the results.

Available tools:

* `inspect_dataset`
* `compare_target_candidates`
* `analyze_target`
* `train_classifier`
* `train_regressor`

---

## 13. Vision Agent

For image-related tasks, the Manager Agent delegates the task to the Vision Agent.

The Vision Agent uses:

* `inspect_image`

to inspect and analyze the uploaded image.

The result is returned to the Manager Agent.

---

## 14. Report Agent

When a report is required, the Manager Agent delegates report generation to the Report Agent.

The Report Agent:

1. Receives the analysis results.
2. Organizes the findings.
3. Generates a professional PDF report.
4. Saves the report.
5. Returns the report information.

The report-generation tool is:

* `create_report_pdf`

---

## 15. View the Execution Results

After the task finishes, the Streamlit frontend displays the execution results.

These can include:

* Request ID
* Agent execution history
* Agent results
* Completion status
* Generated report

The execution history shows how the task was processed by the agent system.

---

## 16. Access the Generated Report

When report generation is requested, the Report Agent creates a PDF report.

Generated reports are stored in:

```text
reports/
```

Uploaded files are handled through:

```text
uploads/
```

The generated report can also be accessed from the Streamlit frontend.

---

# Testing

## 17. Run Tests

Activate the virtual environment:

```bash
source .venv/bin/activate
```

Run the tests:

```bash
pytest
```

---

# Running the Project Again

After the initial setup, you do not need to:

* Clone the repository again
* Create the virtual environment again
* Install dependencies again
* Create the `.env` file again

Simply start the backend and frontend.

### Terminal 1 — Backend

```bash
cd autonomous-task-agent
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

### Terminal 2 — Frontend

```bash
cd autonomous-task-agent
source .venv/bin/activate
streamlit run ui.py
```

Then open:

**[http://localhost:8501](http://localhost:8501)**

---

# Stopping the Application

## 18. Stop the Backend

In the terminal running FastAPI, press:

```text
Ctrl + C
```

---

## 19. Stop the Frontend

In the terminal running Streamlit, press:

```text
Ctrl + C
```

---

## 20. Deactivate the Virtual Environment

After stopping the application:

```bash
deactivate
```

---

# Quick Start

For subsequent runs, open two terminals.

### Terminal 1 — Backend

```bash
cd autonomous-task-agent
source .venv/bin/activate
uvicorn app.api.main:app --reload
```

### Terminal 2 — Frontend

```bash
cd autonomous-task-agent
source .venv/bin/activate
streamlit run ui.py
```

Open the frontend:

**[http://localhost:8501](http://localhost:8501)**

Backend API documentation:

**[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

The application is now ready to execute autonomous tasks.

