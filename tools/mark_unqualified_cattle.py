#!/usr/bin/env python3
"""Mark user-selected cattle as excluded from downstream analysis."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_COW_IDS = {5, 19, 115, 147}
TARGETS = [
    ROOT / "cow_visual_annotations.csv",
    ROOT / "cattle_3d_extraction_66" / "long_axis_records.csv",
]


def mark(path: Path) -> tuple[int, list[int]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    found: list[int] = []
    changed = 0
    for row in rows:
        cow_id = int(row["cow_id"])
        if cow_id not in TARGET_COW_IDS:
            continue
        found.append(cow_id)
        desired = {
            "analysis_data_quality_code": "unqualified",
            "analysis_include": "FALSE",
            "analysis_data_quality_label_zh": "不合格",
            "analysis_data_quality_source": "user_review",
            "analysis_data_quality_annotated_on": "2026-09-05",
            "analysis_data_quality_notes": "用户指定为不合格数据，后续分析排除",
        }
        if any(row.get(key) != value for key, value in desired.items()):
            row.update(desired)
            changed += 1

    missing = sorted(TARGET_COW_IDS - set(found))
    if missing:
        raise ValueError(f"missing cow IDs in {path}: {missing}")

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return changed, sorted(found)


def main() -> None:
    for target in TARGETS:
        changed, found = mark(target)
        print(f"{target.relative_to(ROOT)}: marked {changed} rows; cows={found}")


if __name__ == "__main__":
    main()
