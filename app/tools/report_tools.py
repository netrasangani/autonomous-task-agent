from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def create_report_pdf(
    goal: str,
    report_content: str,
    request_id: str,
    results: list[dict],
) -> dict:
    """
    Create a PDF containing the final autonomous-agent report
    and execution summary.
    """

    output_path = REPORTS_DIR / f"task_report_{request_id}.pdf"

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=8,
    )

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    elements = []

    elements.append(
        Paragraph(
            "Autonomous Task Execution Report",
            title_style,
        )
    )

    elements.append(
        Paragraph("Task Goal", heading_style)
    )

    elements.append(
        Paragraph(
            _escape(goal),
            body_style,
        )
    )

    elements.append(
        Paragraph("Agent Analysis", heading_style)
    )

    for line in report_content.splitlines():
        line = line.strip()

        if not line:
            elements.append(Spacer(1, 6))
            continue

        if line.startswith("#"):
            elements.append(
                Paragraph(
                    _escape(line.lstrip("#").strip()),
                    heading_style,
                )
            )
        else:
            elements.append(
                Paragraph(
                    _escape(line),
                    body_style,
                )
            )

    elements.append(
        Paragraph(
            "Execution Summary",
            heading_style,
        )
    )

    table_data = [["Agent", "Action"]]

    for item in results:
        table_data.append(
            [
                str(item.get("agent", "Unknown")),
                str(item.get("action", "Unknown")),
            ]
        )

    if len(table_data) == 1:
        table_data.append(
            ["None", "No recorded actions."],
        )

    table = Table(
        table_data,
        colWidths=[1.5 * inch, 4.8 * inch],
        repeatRows=1,
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    elements.append(table)
    elements.append(Spacer(1, 15))

    elements.append(
        Paragraph(
            f"Request ID: {_escape(request_id)}",
            body_style,
        )
    )

    document.build(elements)

    return {
        "status": "success",
        "report_path": str(output_path),
        "request_id": request_id,
    }


def _escape(text: str) -> str:
    """Escape special characters for ReportLab paragraphs."""

    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )