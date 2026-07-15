from __future__ import annotations

import csv
from pathlib import Path


HEADER_PREFIX = "method_name,dataset_name,run_id,status,exit_code"


def load_blocks(lines: list[str]) -> list[list[dict[str, str]]]:
    header_indexes = [index for index, line in enumerate(lines) if line.startswith(HEADER_PREFIX)]
    if not header_indexes:
        raise SystemExit("No encontre ningun header valido en all_runs.csv")

    blocks: list[list[dict[str, str]]] = []
    for start, end in zip(header_indexes, header_indexes[1:] + [len(lines)]):
        block_lines = lines[start:end]
        reader = csv.DictReader(block_lines)
        block_rows = list(reader)
        if block_rows:
            blocks.append(block_rows)
    return blocks


def merge_rows(blocks: list[list[dict[str, str]]]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    seen_fields: set[str] = set()
    merged_rows: list[dict[str, str]] = []

    for rows in blocks:
        for row in rows:
            if row.get("method_name") == "method_name":
                continue
            cleaned_row = {
                key: value
                for key, value in row.items()
                if key is not None and value is not None
            }
            merged_rows.append(cleaned_row)
            for key in cleaned_row:
                if key not in seen_fields:
                    seen_fields.add(key)
                    fieldnames.append(key)

    return fieldnames, merged_rows


def rewrite_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    csv_path = Path(__file__).resolve().parent / "results" / "all_runs.csv"
    backup_path = csv_path.with_suffix(".csv.bak")

    lines = csv_path.read_text(encoding="utf-8").splitlines()
    blocks = load_blocks(lines)
    fieldnames, rows = merge_rows(blocks)

    backup_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    rewrite_csv(csv_path, fieldnames, rows)

    print(f"Backup: {backup_path}")
    print(f"Rows limpias: {len(rows)}")
    print(f"Columnas unificadas: {len(fieldnames)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
