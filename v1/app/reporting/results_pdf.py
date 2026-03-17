from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _clean_text(value: Any) -> str:
    if value is None:
        return "N/A"
    s = str(value).strip()
    return s if s else "N/A"


def _contract_id_from_results_path(results_path: Path) -> str:
    return results_path.parent.name


def build_results_pdf(results_jsonl_path: Path, output_pdf_path: Optional[Path] = None) -> Path:
    if not results_jsonl_path.exists():
        raise FileNotFoundError(f"results.jsonl not found: {results_jsonl_path}")

    rows = load_jsonl(results_jsonl_path)
    contract_id = _contract_id_from_results_path(results_jsonl_path)

    if output_pdf_path is None:
        output_pdf_path = results_jsonl_path.parent / f"{contract_id}.pdf"

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"Contract Q&A Results - {contract_id}",
        author="IMC Contract Intelligence Assistant",
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleCustom",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1f3b5b"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )

    meta_style = ParagraphStyle(
        "MetaCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#5b6b7a"),
        spaceAfter=10,
    )

    q_style = ParagraphStyle(
        "QuestionCustom",
        parent=styles["Heading4"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#153e75"),
        spaceBefore=6,
        spaceAfter=4,
    )

    a_label_style = ParagraphStyle(
        "AnswerLabelCustom",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor("#334e68"),
        spaceAfter=2,
    )

    a_style = ParagraphStyle(
        "AnswerCustom",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.black,
        spaceAfter=10,
    )

    story = []
    story.append(Paragraph("Contract Questions and Answers", title_style))
    story.append(Paragraph(f"Contract ID: {contract_id}<br/>Total Questions: {len(rows)}", meta_style))
    story.append(Spacer(1, 4))

    for idx, row in enumerate(rows, start=1):
        question = _clean_text(row.get("question"))
        value = _clean_text(row.get("value"))

        story.append(Paragraph(f"{idx}. {question}", q_style))
        story.append(Paragraph("Answer", a_label_style))
        story.append(Paragraph(value.replace("\n", "<br/>"), a_style))
        story.append(Spacer(1, 4))

    doc.build(story)
    return output_pdf_path