#!/usr/bin/env python3
"""Restore all 66 selected cattle to the manually included analysis set."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "cow_visual_annotations.csv",
    ROOT / "cattle_3d_extraction_66" / "long_axis_records.csv",
]


def restore(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed = 0
    for row in rows:
        if row.get("analysis_data_quality_code") == "unqualified" or row.get("analysis_include") == "FALSE":
            row["analysis_data_quality_code"] = "qualified"
            row["analysis_include"] = "TRUE"
            row["analysis_data_quality_label_zh"] = "合格"
            row["analysis_data_quality_source"] = "user_review"
            row["analysis_data_quality_annotated_on"] = "2026-09-05"
            row["analysis_data_quality_notes"] = ""
            changed += 1

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed


def main() -> None:
    for target in TARGETS:
        print(f"{target.relative_to(ROOT)}: restored {restore(target)} rows")


if __name__ == "__main__":
    main()
