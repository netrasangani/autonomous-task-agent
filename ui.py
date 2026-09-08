from pathlib import Path

import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/run"

st.set_page_config(
    page_title="Autonomous Task Execution Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.75;
        margin-bottom: 25px;
    }

    .agent-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 10px;
    }

    .agent-title {
        font-size: 20px;
        font-weight: 600;
    }

    .agent-description {
        font-size: 14px;
        opacity: 0.75;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 Autonomous Task Execution Agent</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    An agentic AI system that understands a high-level goal,
    dynamically selects specialist agents, executes tools,
    evaluates results, and produces a final report.
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR — SYSTEM ARCHITECTURE
# =========================================================

with st.sidebar:

    st.header("🏗️ System Architecture")

    st.markdown(
        """
        **1️⃣ Manager Agent**

        Decides what should happen next.

        ↓

        **2️⃣ Specialist Agents**

        📊 Data & ML Agent  
        👁️ Vision Agent  
        📄 Report Agent

        ↓

        **3️⃣ Tools**

        Actual Python functions perform
        data analysis, ML, CV, and reporting.

        ↓

        **4️⃣ Final Output**

        Results + findings + PDF report
        """
    )

    st.divider()

    st.subheader("🧠 Agent Roles")

    st.markdown(
        """
        **Manager Agent**  
        Autonomous decision-maker.

        **DataAgent**  
        Dataset analysis and machine learning.

        **VisionAgent**  
        Computer vision and image analysis.

        **ReportAgent**  
        Final synthesis and PDF generation.
        """
    )

    st.divider()

    st.caption(
        "FastAPI Backend • OpenAI Responses API • "
        "Python Tools • Streamlit UI"
    )


# =========================================================
# AGENT OVERVIEW
# =========================================================

st.markdown(
    '<div class="section-title">🧩 Available Agents</div>',
    unsafe_allow_html=True,
)

agent_col1, agent_col2, agent_col3 = st.columns(3)


with agent_col1:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-title">🧠 Manager Agent</div>

        <div class="agent-description">
        Analyzes the current task state and dynamically
        decides which specialist should act next.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with agent_col2:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-title">📊 DataAgent</div>

        <div class="agent-description">
        Performs dataset inspection, target analysis,
        preprocessing, model training and evaluation.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


with agent_col3:

    st.markdown(
        """
        <div class="agent-card">

        <div class="agent-title">👁️ VisionAgent</div>

        <div class="agent-description">
        Performs image inspection and computer
        vision-related tasks when required.
        </div>

        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# TASK INPUT
# =========================================================

st.divider()

st.markdown(
    '<div class="section-title">🎯 Define Your Task</div>',
    unsafe_allow_html=True,
)

goal = st.text_area(
    "What do you want the autonomous agent to accomplish?",
    height=150,
    placeholder=(
        "Example:\n"
        "Analyze this dataset, determine the most appropriate "
        "machine learning approach, train and evaluate suitable "
        "models, interpret the results, and generate a professional "
        "PDF report."
    ),
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-title">📁 Provide Input</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a CSV dataset or image",
    type=[
        "csv",
        "png",
        "jpg",
        "jpeg",
        "webp",
    ],
    help=(
        "CSV files can be used for data analysis and ML. "
        "Images can be used by the VisionAgent."
    ),
)


# Show uploaded file information

if uploaded_file is not None:

    file_size_kb = len(
        uploaded_file.getvalue()
    ) / 1024

    file_col1, file_col2, file_col3 = st.columns(3)

    with file_col1:
        st.metric(
            "File",
            uploaded_file.name,
        )

    with file_col2:
        st.metric(
            "Type",
            uploaded_file.type,
        )

    with file_col3:
        st.metric(
            "Size",
            f"{file_size_kb:.1f} KB",
        )


# =========================================================
# EXECUTION BUTTON
# =========================================================

st.divider()

run_button = st.button(
    "🚀 Run Autonomous Agent",
    type="primary",
    use_container_width=True,
)


# =========================================================
# EXECUTION
# =========================================================

if run_button:

    # -----------------------------------------------------
    # Validate goal
    # -----------------------------------------------------

    if not goal.strip():

        st.error(
            "❌ Please enter a task goal before running the agent."
        )

        st.stop()


    # -----------------------------------------------------
    # Prepare uploaded file
    # -----------------------------------------------------

    files = None

    if uploaded_file is not None:

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )
        }


    # -----------------------------------------------------
    # Show execution status
    # -----------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-title">⚙️ Autonomous Execution</div>',
        unsafe_allow_html=True,
    )

    status_placeholder = st.empty()

    status_placeholder.info(
        "🧠 Manager Agent is analyzing the task..."
    )


    # -----------------------------------------------------
    # Call FastAPI backend
    # -----------------------------------------------------

    try:

        response = requests.post(
            API_URL,
            data={
                "goal": goal.strip()
            },
            files=files,
            timeout=300,
        )

    except requests.exceptions.ConnectionError:

        status_placeholder.empty()

        st.error(
            """
            ❌ Could not connect to the FastAPI backend.

            Make sure your backend is running with:

            `python run.py`
            """
        )

        st.stop()

    except requests.exceptions.Timeout:

        status_placeholder.empty()

        st.error(
            "⏳ The agent took too long to complete."
        )

        st.stop()

    except Exception as exc:

        status_placeholder.empty()

        st.error(
            f"❌ Unexpected error: {exc}"
        )

        st.stop()


    # -----------------------------------------------------
    # Handle API errors
    # -----------------------------------------------------

    if response.status_code != 200:

        status_placeholder.empty()

        st.error(
            "❌ Agent execution failed."
        )

        with st.expander(
            "View backend error"
        ):

            st.code(
                response.text
            )

        st.stop()


    # -----------------------------------------------------
    # Parse response
    # -----------------------------------------------------

    result = response.json()

    status_placeholder.success(
        "✅ Autonomous execution completed."
    )


    # =====================================================
    # EXECUTION SUMMARY
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📌 Execution Summary</div>',
        unsafe_allow_html=True,
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = (
        st.columns(4)
    )


    with summary_col1:

        st.metric(
            "Status",
            (
                "Completed"
                if result.get("completed")
                else "Incomplete"
            ),
        )


    with summary_col2:

        st.metric(
            "Manager Iterations",
            result.get(
                "iterations",
                0,
            ),
        )


    with summary_col3:

        st.metric(
            "Agent Actions",
            len(
                result.get(
                    "history",
                    [],
                )
            ),
        )


    with summary_col4:

        request_id = result.get(
            "request_id",
            "",
        )

        st.metric(
            "Request ID",
            (
                request_id[:8] + "..."
                if request_id
                else "N/A"
            ),
        )


    # =====================================================
    # AGENT EXECUTION TRACE
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🔄 Agent Execution Trace</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "The Manager Agent dynamically selected the actions "
        "shown below during execution."
    )

    history = result.get(
        "history",
        [],
    )


    if history:

        for index, item in enumerate(
            history,
            start=1,
        ):

            agent = item.get(
                "agent",
                "Unknown Agent",
            )

            action = item.get(
                "action",
                "No action description available.",
            )

            if agent == "DataAgent":

                icon = "📊"

            elif agent == "VisionAgent":

                icon = "👁️"

            elif agent == "ReportAgent":

                icon = "📄"

            else:

                icon = "🤖"


            with st.expander(
                f"{icon} Step {index} — {agent}",
                expanded=True,
            ):

                st.write(
                    action
                )

    else:

        st.info(
            "No execution trace was recorded."
        )


    # =====================================================
    # FINAL RESULT
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🧠 Final Result</div>',
        unsafe_allow_html=True,
    )

    final_result = result.get(
        "final_result",
        "No final result was returned.",
    )

    st.success(
        final_result
    )


    # =====================================================
    # DETAILED AGENT RESULTS
    # =====================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Detailed Results</div>',
        unsafe_allow_html=True,
    )

    results = result.get(
        "results",
        [],
    )


    if results:

        for index, item in enumerate(
            results,
            start=1,
        ):

            agent = item.get(
                "agent",
                "Unknown Agent",
            )

            action = item.get(
                "action",
                "",
            )

            item_result = item.get(
                "result",
                {},
            )


            with st.expander(
                f"{index}. {agent}",
                expanded=False,
            ):

                st.markdown(
                    "**Task performed:**"
                )

                st.write(
                    action
                )

                st.markdown(
                    "**Agent output:**"
                )


                if isinstance(
                    item_result,
                    dict,
                ):

                    st.json(
                        item_result
                    )

                else:

                    st.write(
                        item_result
                    )

    else:

        st.info(
            "No detailed results were returned."
        )


    # =====================================================
    # PDF REPORT
    # =====================================================

    report_path = None

    for item in results:

        item_result = item.get(
            "result"
        )

        if isinstance(
            item_result,
            dict,
        ):

            pdf_data = item_result.get(
                "pdf"
            )

            if isinstance(
                pdf_data,
                dict,
            ):

                report_path = pdf_data.get(
                    "report_path"
                )


    if report_path:

        st.divider()

        st.markdown(
            '<div class="section-title">📄 Generated Report</div>',
            unsafe_allow_html=True,
        )

        report_file = Path(
            report_path
        )


        if report_file.exists():

            st.success(
                "✅ Professional PDF report generated successfully."
            )

            st.download_button(
                label="⬇️ Download PDF Report",
                data=report_file.read_bytes(),
                file_name=report_file.name,
                mime="application/pdf",
                use_container_width=True,
            )

        else:

            st.warning(
                f"Report generated at: {report_path}"
            )


    # =====================================================
    # TASK INFORMATION
    # =====================================================

    st.divider()

    with st.expander(
        "🔍 View Task Information"
    ):

        st.write(
            "**Goal:**"
        )

        st.write(
            result.get(
                "goal",
                goal,
            )
        )

        st.write(
            "**Input File:**"
        )

        st.write(
            result.get(
                "file_path",
                "No file",
            )
        )

        st.write(
            "**Request ID:**"
        )

        st.code(
            result.get(
                "request_id",
                "N/A",
            )
        )