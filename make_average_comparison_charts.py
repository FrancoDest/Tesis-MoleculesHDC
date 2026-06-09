from __future__ import annotations

import csv
import argparse
import re
from pathlib import Path
from typing import Dict, List


FINAL_DIR = Path(__file__).resolve().parent
DEFAULT_ALL_RUNS_CSV = FINAL_DIR / "results" / "all_runs.csv"
DEFAULT_OUTPUT_DIR = FINAL_DIR / "results" / "average_comparisons"

METHOD_LABELS = {
    "random_forest_rdkit": "Random Forest RDKit",
    "mole_bert_hdc": "Mole-BERT HDC",
    "mol2vec_jl": "Mol2Vec JL",
    "molehd": "MoleHD",
    "graphhd": "GraphHD",
    "graphHD": "GraphHD",
    "polymer_gnn_bbbp": "Polymer GNN",
}

METHOD_COLORS = {
    "random_forest_rdkit": "#1d3557",
    "mole_bert_hdc": "#e76f51",
    "mol2vec_jl": "#2a9d8f",
    "molehd": "#7b2cbf",
    "graphhd": "#c1121f",
    "graphHD": "#c1121f",
    "polymer_gnn_bbbp": "#6c757d",
}

DEFAULT_METHOD_COLOR = "#6c757d"

CORE_METRICS = [
    ("accuracy", "Accuracy"),
    ("auroc", "AUROC"),
    ("balanced_accuracy", "Balanced Accuracy"),
    ("f1", "F1"),
]

TRADEOFF_METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
]

ALL_METRICS = CORE_METRICS + TRADEOFF_METRICS
DATASET_SIZES = {
    "mutag": 188,
    "bbbp": 2039,
}


def to_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def preferred_metric(row: dict, average_key: str, log_key: str) -> float | None:
    return to_float(row.get(average_key)) or to_float(row.get(log_key))


def read_log_text(run_log_path: str | None) -> str | None:
    if not run_log_path:
        return None
    path = Path(run_log_path)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def parse_metrics_block(text: str) -> Dict[str, float]:
    patterns = {
        "accuracy": [r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)"],
        "auroc": [r"Auroc:\s*([0-9]+(?:\.[0-9]+)?)", r"AUROC:\s*([0-9]+(?:\.[0-9]+)?)"],
        "balanced_accuracy": [r"Bacc:\s*([0-9]+(?:\.[0-9]+)?)"],
        "f1": [r"F1:\s*([0-9]+(?:\.[0-9]+)?)"],
        "precision": [r"Precision:\s*([0-9]+(?:\.[0-9]+)?)"],
        "recall": [r"Recall:\s*([0-9]+(?:\.[0-9]+)?)"],
    }
    parsed: Dict[str, float] = {}
    for metric_name, metric_patterns in patterns.items():
        for pattern in metric_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                parsed[metric_name] = float(match.group(1))
                break
    return parsed


def extract_average_block(text: str) -> str | None:
    header_patterns = [
        r"Average Stats for .*?iterations",
        r"Average metrics:",
        r"Promedio para .*?corridas",
    ]
    for header_pattern in header_patterns:
        match = re.search(header_pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        remainder = text[match.end():]
        end_markers = [
            r"\n\s*\n(?:Stats corresponding|Stats correspondientes|Tiempos totales:|Average CPU Total Time:|Average Memory Usage:|Peak Memory Usage:)",
            r"\nTiempos totales:",
            r"\nAverage CPU Total Time:",
            r"\nAverage Memory Usage:",
            r"\nPeak Memory Usage:",
        ]
        end_positions = []
        for end_marker in end_markers:
            end_match = re.search(end_marker, remainder, flags=re.IGNORECASE)
            if end_match:
                end_positions.append(end_match.start())
        end = min(end_positions) if end_positions else len(remainder)
        return remainder[:end]
    return None


def parse_average_metrics_from_log(run_log_path: str | None) -> Dict[str, float]:
    text = read_log_text(run_log_path)
    if not text:
        return {}
    block = extract_average_block(text)
    if not block:
        return {}
    return parse_metrics_block(block)


def parse_fallback_metric_from_log(run_log_path: str | None, metric_name: str) -> float | None:
    text = read_log_text(run_log_path)
    if not text:
        return None

    patterns = {
        "accuracy": [r"Accuracy:\s*([0-9]+(?:\.[0-9]+)?)"],
        "auroc": [r"Auroc:\s*([0-9]+(?:\.[0-9]+)?)", r"AUROC:\s*([0-9]+(?:\.[0-9]+)?)"],
        "balanced_accuracy": [r"Bacc:\s*([0-9]+(?:\.[0-9]+)?)"],
        "f1": [r"F1:\s*([0-9]+(?:\.[0-9]+)?)"],
        "precision": [r"Precision:\s*([0-9]+(?:\.[0-9]+)?)"],
        "recall": [r"Recall:\s*([0-9]+(?:\.[0-9]+)?)"],
    }
    for pattern in patterns.get(metric_name, []):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Genera gráficas comparativas promedio a partir de final/results/all_runs.csv."
    )
    parser.add_argument(
        "--all-runs-csv",
        type=Path,
        default=DEFAULT_ALL_RUNS_CSV,
        help="CSV consolidado de corridas. Default: final/results/all_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta de salida para SVGs y README. Default: final/results/average_comparisons",
    )
    return parser.parse_args()


def load_runs(all_runs_csv: Path) -> Dict[str, List[dict]]:
    with all_runs_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    runs: Dict[tuple[str, str, str], dict] = {}
    for row in rows:
        method = row["method_name"]
        dataset = row["dataset_name"]
        run_id = row["run_id"]
        key = (method, dataset, run_id)
        merged = runs.setdefault(
            key,
            {
                "method_name": method,
                "dataset_name": dataset,
                "run_id": run_id,
                "status": row.get("status", ""),
            },
        )
        merged["status"] = merged["status"] or row.get("status", "")
        merged["run_log"] = merged.get("run_log") or row.get("run_log", "")

        metric_map = {
            "accuracy": (
                to_float(row.get("average_accuracy"))
                or to_float(row.get("average_Accuracy"))
            ),
            "auroc": (
                to_float(row.get("average_roc_auc"))
                or to_float(row.get("average_AUROC"))
                or to_float(row.get("average_AUROC_OVR_Macro"))
            ),
            "balanced_accuracy": (
                to_float(row.get("average_balanced_accuracy"))
                or to_float(row.get("average_Balanced_Accuracy"))
            ),
            "f1": (
                to_float(row.get("average_f1"))
                or to_float(row.get("average_F1"))
                or to_float(row.get("average_F1_Score"))
                or to_float(row.get("average_F1_Macro"))
            ),
            "precision": (
                to_float(row.get("average_precision"))
                or to_float(row.get("average_Precision"))
                or to_float(row.get("average_Precision_Macro"))
            ),
            "recall": (
                to_float(row.get("average_recall"))
                or to_float(row.get("average_Recall"))
                or to_float(row.get("average_Recall_Macro"))
            ),
        }
        for metric_name, value in metric_map.items():
            if value is not None:
                merged[metric_name] = value

    for merged in runs.values():
        average_metrics = parse_average_metrics_from_log(merged.get("run_log"))
        for metric_name, value in average_metrics.items():
            merged[metric_name] = value
        for metric_name, _label in ALL_METRICS:
            if merged.get(metric_name) is None:
                fallback = {
                    "accuracy": to_float(merged.get("log_accuracy")),
                    "auroc": to_float(merged.get("log_auroc")),
                    "balanced_accuracy": to_float(merged.get("log_bacc")),
                    "f1": to_float(merged.get("log_f1")),
                    "precision": to_float(merged.get("log_precision")),
                    "recall": to_float(merged.get("log_recall")),
                }.get(metric_name)
                if fallback is not None:
                    merged[metric_name] = fallback
            if merged.get(metric_name) is None:
                parsed = parse_fallback_metric_from_log(merged.get("run_log"), metric_name)
                if parsed is not None:
                    merged[metric_name] = parsed

    by_dataset: Dict[str, List[dict]] = {}
    for merged in runs.values():
        if not any(metric in merged for metric, _ in ALL_METRICS):
            continue
        by_dataset.setdefault(merged["dataset_name"], []).append(merged)

    selected: Dict[str, List[dict]] = {}
    for dataset, items in by_dataset.items():
        latest_by_method: Dict[str, dict] = {}
        for item in sorted(items, key=lambda it: it["run_id"]):
            latest_by_method[item["method_name"]] = item
        selected[dataset] = list(latest_by_method.values())
    return selected


def save_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_name",
        "method_name",
        "method_label",
        "status",
        "run_id",
        "accuracy",
        "auroc",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "normal", fill: str = "#111111") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-family="Arial, sans-serif" font-weight="{weight}" fill="{fill}">{safe}</text>'


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "Sin dato"
    return f"{value:.2f}"


def method_color(method_name: str) -> str:
    return METHOD_COLORS.get(method_name, DEFAULT_METHOD_COLOR)


def draw_panel(parts: List[str], rows: List[dict], metric_key: str, metric_label: str, x0: float, y0: float, w: float, h: float) -> None:
    parts.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{w:.1f}" height="{h:.1f}" rx="18" fill="#ffffff" stroke="#e5e7eb"/>')
    parts.append(svg_text(x0 + 18, y0 + 30, metric_label, 16, weight="bold"))

    inner_left = x0 + 18
    inner_right = x0 + w - 18
    inner_top = y0 + 52
    inner_bottom = y0 + h - 18
    bar_area_w = inner_right - inner_left - 170
    row_h = (inner_bottom - inner_top) / max(len(rows), 1)

    for tick in range(6):
        value = tick / 5
        x = inner_left + 150 + value * bar_area_w
        parts.append(f'<line x1="{x:.1f}" y1="{inner_top:.1f}" x2="{x:.1f}" y2="{inner_bottom:.1f}" stroke="#f1f5f9" stroke-width="1"/>')
        parts.append(svg_text(x, inner_bottom + 16, f"{value:.1f}", 10, anchor="middle", fill="#777777"))

    ordered = sorted(rows, key=lambda row: (-1 if row.get(metric_key) is None else -row[metric_key], row["method_label"]))
    for idx, row in enumerate(ordered):
        y = inner_top + idx * row_h + row_h * 0.65
        value = row.get(metric_key)
        label = row["method_label"] + (" *" if row["status"] != "ok" else "")
        parts.append(svg_text(inner_left, y, label, 12, weight="bold"))
        bar_x = inner_left + 150
        if value is None:
            parts.append(f'<rect x="{bar_x:.1f}" y="{y - 10:.1f}" width="{bar_area_w:.1f}" height="18" rx="9" fill="#f3f4f6"/>')
            parts.append(svg_text(bar_x + 8, y + 1, "Sin dato", 11, fill="#777777"))
            continue
        fill_w = bar_area_w * value
        color = method_color(row["method_name"])
        parts.append(f'<rect x="{bar_x:.1f}" y="{y - 10:.1f}" width="{bar_area_w:.1f}" height="18" rx="9" fill="#edf2f7"/>')
        parts.append(f'<rect x="{bar_x:.1f}" y="{y - 10:.1f}" width="{fill_w:.1f}" height="18" rx="9" fill="{color}"/>')
        parts.append(svg_text(bar_x + bar_area_w + 8, y + 1, fmt_metric(value), 11, weight="bold"))


def render_small_multiples(dataset: str, rows: List[dict], metrics: List[tuple[str, str]], out_path: Path, title: str, subtitle: str) -> None:
    width = 1320
    height = 920 if len(metrics) == 4 else 560
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        svg_text(40, 42, f"{dataset.upper()} - {title}", 28, weight="bold"),
        svg_text(40, 68, subtitle, 14, fill="#555555"),
    ]

    if len(metrics) == 4:
        panel_w = 600
        panel_h = 340
        positions = [(40, 96), (680, 96), (40, 470), (680, 470)]
    else:
        panel_w = 600
        panel_h = 340
        positions = [(40, 140), (680, 140)]

    for (metric_key, metric_label), (x, y) in zip(metrics, positions):
        draw_panel(parts, rows, metric_key, metric_label, x, y, panel_w, panel_h)

    parts.append(svg_text(40, height - 18, "* Hay métricas reportadas pero la corrida quedó marcada como failed; se muestra igual para no perder información útil.", 12, fill="#8a4b08"))
    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def render_dataset_scale_chart(all_rows: List[dict], out_path: Path) -> None:
    width = 1220
    height = 760
    left = 120
    right = 80
    top = 100
    bottom = 110
    chart_w = width - left - right
    chart_h = height - top - bottom

    mutag_x = left
    bbbp_x = left + chart_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fffdf8"/>',
        svg_text(40, 42, "Dataset Size vs AUROC", 28, weight="bold"),
        svg_text(40, 68, "Cada método une su AUROC promedio entre MUTAG (188 moléculas) y BBBP (2039 moléculas).", 14, fill="#555555"),
    ]

    for tick in range(6):
        value = tick / 5
        y = top + chart_h - value * chart_h
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + chart_w}" y2="{y:.1f}" stroke="#eef2f7" stroke-width="1"/>')
        parts.append(svg_text(left - 12, y + 4, f"{value:.1f}", 11, anchor="end", fill="#666666"))

    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#222222" stroke-width="1.5"/>')
    parts.append(f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" y2="{top + chart_h}" stroke="#222222" stroke-width="1.5"/>')
    parts.append(svg_text(mutag_x, top + chart_h + 26, "MUTAG", 13, anchor="middle", weight="bold"))
    parts.append(svg_text(mutag_x, top + chart_h + 44, "188 moleculas", 12, anchor="middle", fill="#555555"))
    parts.append(svg_text(bbbp_x, top + chart_h + 26, "BBBP", 13, anchor="middle", weight="bold"))
    parts.append(svg_text(bbbp_x, top + chart_h + 44, "2039 moleculas", 12, anchor="middle", fill="#555555"))

    rows_by_method: Dict[str, Dict[str, dict]] = {}
    for row in all_rows:
        rows_by_method.setdefault(row["method_name"], {})[row["dataset_name"]] = row

    legend_items = []
    for method_name, per_dataset in sorted(rows_by_method.items(), key=lambda item: METHOD_LABELS.get(item[0], item[0]).lower()):
        mutag = per_dataset.get("mutag")
        bbbp = per_dataset.get("bbbp")
        if not mutag or not bbbp:
            continue
        mutag_auroc = mutag.get("auroc")
        bbbp_auroc = bbbp.get("auroc")
        if mutag_auroc is None or bbbp_auroc is None:
            continue
        color = method_color(method_name)
        y1 = top + chart_h - mutag_auroc * chart_h
        y2 = top + chart_h - bbbp_auroc * chart_h
        parts.append(f'<line x1="{mutag_x:.1f}" y1="{y1:.1f}" x2="{bbbp_x:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{mutag_x:.1f}" cy="{y1:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(f'<circle cx="{bbbp_x:.1f}" cy="{y2:.1f}" r="7" fill="{color}" stroke="#ffffff" stroke-width="2"/>')
        parts.append(svg_text(mutag_x - 12, y1 - 10, fmt_metric(mutag_auroc), 11, anchor="end", weight="bold"))
        parts.append(svg_text(bbbp_x + 12, y2 - 10, fmt_metric(bbbp_auroc), 11, anchor="start", weight="bold"))
        legend_items.append((method_name, METHOD_LABELS.get(method_name, method_name)))

    legend_x = 40
    legend_y = height - 70
    for idx, (method_name, label) in enumerate(legend_items):
        x = legend_x + (idx % 3) * 360
        y = legend_y + (idx // 3) * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 18}" y2="{y}" stroke="{method_color(method_name)}" stroke-width="4" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x + 9}" cy="{y}" r="5" fill="{method_color(method_name)}" stroke="#ffffff" stroke-width="1.5"/>')
        parts.append(svg_text(x + 28, y + 4, label, 12))

    parts.append("</svg>")
    out_path.write_text("\n".join(parts), encoding="utf-8")


def build_rank_summary(rows: List[dict], metrics: List[tuple[str, str]]) -> List[str]:
    lines: List[str] = []
    for metric_key, metric_label in metrics:
        available = [row for row in rows if row.get(metric_key) is not None]
        if not available:
            continue
        ordered = sorted(available, key=lambda row: row[metric_key], reverse=True)
        top = ordered[0]
        lines.append(f"- {metric_label}: destaca {top['method_label']} con {fmt_metric(top[metric_key])}.")
    return lines


def write_dashboard(by_dataset: Dict[str, List[dict]], output_dir: Path, all_runs_csv: Path) -> None:
    dashboard = output_dir / "README.md"
    lines = [
        "# Average Comparison Dashboard",
        "",
        f"Comparativas construidas a partir de promedios en `{all_runs_csv}`.",
        "",
        "Reglas:",
        "- Se toma la ultima corrida con métricas por método y dataset.",
        "- Se priorizan columnas `average_*`; si no existen, se usan los promedios reportados en log.",
        "- Si una corrida tiene métricas pero `status != ok`, se marca con `*`.",
        "",
        "## General",
        "",
        "- [Dataset size vs AUROC](./dataset_size_vs_auroc.svg)",
        "",
    ]

    for dataset, rows in sorted(by_dataset.items()):
        lines.extend(
            [
                f"## {dataset.upper()}",
                "",
                f"- [Métricas principales](./{dataset}_main_metrics.svg)",
                f"- [Trade-offs de precision y recall](./{dataset}_tradeoffs.svg)",
                "",
            ]
        )
        lines.extend(build_rank_summary(rows, CORE_METRICS))
        lines.append("")
        lines.extend(build_rank_summary(rows, TRADEOFF_METRICS))
        lines.append("")

    lines.extend(
        [
            "## Datos Consolidados",
            "",
            "- [comparison_summary.csv](./comparison_summary.csv)",
            "",
        ]
    )
    dashboard.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    all_runs_csv = args.all_runs_csv.resolve()

    if not all_runs_csv.exists():
        raise SystemExit(f"No existe el archivo de corridas: {all_runs_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    by_dataset = load_runs(all_runs_csv)

    clean_rows: List[dict] = []
    enriched_by_dataset: Dict[str, List[dict]] = {}
    for dataset, rows in sorted(by_dataset.items()):
        enriched = []
        for row in rows:
            enriched_row = {
                "dataset_name": dataset,
                "method_name": row["method_name"],
                "method_label": METHOD_LABELS.get(row["method_name"], row["method_name"]),
                "status": row["status"],
                "run_id": row["run_id"],
            }
            for metric_key, _ in ALL_METRICS:
                enriched_row[metric_key] = row.get(metric_key)
            enriched.append(enriched_row)
        enriched.sort(key=lambda row: row["method_label"].lower())
        enriched_by_dataset[dataset] = enriched
        clean_rows.extend(enriched)

        render_small_multiples(
            dataset,
            enriched,
            CORE_METRICS,
            output_dir / f"{dataset}_main_metrics.svg",
            title="Métricas Principales",
            subtitle="Un panel por métrica para comparar sin mezclar demasiada información en un solo gráfico.",
        )
        render_small_multiples(
            dataset,
            enriched,
            TRADEOFF_METRICS,
            output_dir / f"{dataset}_tradeoffs.svg",
            title="Trade-offs",
            subtitle="Separado del resto para ver rápido quién prioriza precision y quién recall.",
        )

    save_csv(clean_rows, output_dir / "comparison_summary.csv")
    render_dataset_scale_chart(clean_rows, output_dir / "dataset_size_vs_auroc.svg")
    write_dashboard(enriched_by_dataset, output_dir, all_runs_csv)
    print(f"Comparativas generadas en: {output_dir}")


if __name__ == "__main__":
    main()
