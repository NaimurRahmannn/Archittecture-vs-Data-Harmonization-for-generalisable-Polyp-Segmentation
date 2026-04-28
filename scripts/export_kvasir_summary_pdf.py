from __future__ import annotations

import argparse
import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def parse_args():
    parser = argparse.ArgumentParser(description="Export a thesis-ready Kvasir comparison PDF.")
    parser.add_argument(
        "--valdice_csv",
        default="outputs/kvasir_m2b_valdice_mean_std.csv",
        help="Aggregated valDice CSV.",
    )
    parser.add_argument(
        "--seqval_csv",
        default="outputs/kvasir_m2b_seqval_constrained_mean_std.csv",
        help="Aggregated SeqVal-constrained CSV.",
    )
    parser.add_argument(
        "--out_pdf",
        default="outputs/kvasir_m2b_comparison_summary.pdf",
        help="Output PDF path.",
    )
    return parser.parse_args()


def read_overall_row(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row.get("subset") == "overall":
            return row
    raise ValueError(f"No overall row found in {path}")


def build_paragraph(valdice: dict[str, str], seqval: dict[str, str]) -> str:
    return (
        "Using the aggregated results from the Kvasir external-test runs, we evaluated external domain "
        "generalisation with no retraining using the same M2b ImageNet model family at 256x256 resolution "
        "and the paper-consistent threshold of 0.5. The main checkpoint choice, <b>valDice</b>, achieved "
        f"strong and stable cross-seed performance with Dice {valdice['Dice_mean_std']}, IoU {valdice['IoU_mean_std']}, "
        f"Precision {valdice['Precision_mean_std']}, Recall {valdice['Recall_mean_std']}, F2 {valdice['F2_mean_std']}, "
        f"and area_ratio {valdice['area_ratio_mean_std']}. In contrast, <b>SeqVal-constrained</b> increased "
        f"Recall to {seqval['Recall_mean_std']} and F2 to {seqval['F2_mean_std']}, but reduced Dice, IoU, "
        f"and Precision and increased area_ratio to {seqval['area_ratio_mean_std']}, indicating a more "
        "recall-oriented operating point with moderate mask inflation. Therefore, for the main thesis narrative, "
        "the Kvasir cross-dataset experiment supports the claim that the paper's main M2b ImageNet model "
        "generalises well to an unseen external dataset without retraining, while SeqVal-constrained should "
        "be interpreted as a recall-versus-calibration trade-off rather than uniformly superior external performance."
    )


def build_table_rows(valdice: dict[str, str], seqval: dict[str, str]) -> list[list[str]]:
    return [
        [
            "Selection Strategy",
            "Dice",
            "IoU",
            "Precision",
            "Recall",
            "F2",
            "area_ratio",
            "Interpretation",
        ],
        [
            "valDice",
            valdice["Dice_mean_std"],
            valdice["IoU_mean_std"],
            valdice["Precision_mean_std"],
            valdice["Recall_mean_std"],
            valdice["F2_mean_std"],
            valdice["area_ratio_mean_std"],
            "Best balanced result; main paper-aligned checkpoint choice",
        ],
        [
            "SeqVal-constrained",
            seqval["Dice_mean_std"],
            seqval["IoU_mean_std"],
            seqval["Precision_mean_std"],
            seqval["Recall_mean_std"],
            seqval["F2_mean_std"],
            seqval["area_ratio_mean_std"],
            "Higher recall/F2, but lower Dice/precision and more over-segmentation",
        ],
    ]


def main():
    args = parse_args()
    valdice = read_overall_row(Path(args.valdice_csv))
    seqval = read_overall_row(Path(args.seqval_csv))

    out_pdf = Path(args.out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCenter",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceAfter=10,
    )
    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        spaceAfter=12,
    )
    note_style = ParagraphStyle(
        "Note",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#444444"),
        spaceBefore=8,
    )

    story = [
        Paragraph("Kvasir-SEG Cross-Dataset Summary", title_style),
        Paragraph(
            "Model: M2b ImageNet | Seeds: 0, 1, 2 | Resolution: 256x256 | Threshold: 0.5",
            styles["Heading4"],
        ),
        Spacer(1, 8),
        Paragraph(build_paragraph(valdice, seqval), body_style),
    ]

    table = Table(
        build_table_rows(valdice, seqval),
        colWidths=[1.25 * inch, 0.8 * inch, 0.8 * inch, 0.95 * inch, 0.9 * inch, 0.8 * inch, 0.95 * inch, 2.3 * inch],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#7d8ca3")),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f4f7fb")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f9fbfd"), colors.HexColor("#eef3f8")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (1, 1), (6, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(table)
    story.append(
        Paragraph(
            "Interpretation: valDice is the better main external result, while SeqVal-constrained is the better recall-oriented external result.",
            note_style,
        )
    )

    doc.build(story)
    print(f"saved_pdf={out_pdf.resolve()}")


if __name__ == "__main__":
    main()
