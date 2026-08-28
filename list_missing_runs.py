"""Audita results/ contra catalog.json: que combinaciones (metodo, dataset)
todavia no tienen una corrida exitosa. No ejecuta nada -- solo informa.

Uso:
    python list_missing_runs.py
    python list_missing_runs.py --write-pairs pending_pairs.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

from catalog_config import DEFAULT_CATALOG, DEFAULT_RESULTS_DIR
from pending_runs import compute_all_combo_statuses, pending_combos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--write-pairs",
        type=Path,
        default=None,
        help="Si se pasa, escribe ahi una linea 'metodo,dataset' por cada combo pendiente (missing+failed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    statuses = compute_all_combo_statuses(catalog_path=args.catalog, results_dir=args.results_dir)

    missing = [s for s in statuses if s.state == "missing"]
    failed = [s for s in statuses if s.state == "failed"]
    ok = [s for s in statuses if s.state == "ok"]

    print(f"Total combos (metodo, dataset) en el catalogo: {len(statuses)}")
    print(f"  OK (ultima corrida exitosa): {len(ok)}")
    print(f"  Nunca corridos:              {len(missing)}")
    print(f"  Con la ultima corrida fallida: {len(failed)}")
    print()

    if missing:
        print("=== Nunca corridos ===")
        for combo in missing:
            print(f"  {combo.method_name:32s} {combo.dataset_name}")
        print()

    if failed:
        print("=== Ultima corrida fallida (candidatos a re-generar) ===")
        for combo in failed:
            print(f"  {combo.method_name:32s} {combo.dataset_name:20s} run_id={combo.last_run_id} status={combo.last_status}")
        print()

    print("=== OK, con el run_id de la corrida mas reciente (por si alguno esta viejo/pre-fix) ===")
    for combo in sorted(ok, key=lambda c: c.last_run_id or ""):
        print(f"  {combo.method_name:32s} {combo.dataset_name:20s} run_id={combo.last_run_id}")

    if args.write_pairs:
        pending = pending_combos(statuses)
        with args.write_pairs.open("w", encoding="utf-8") as handle:
            for combo in pending:
                handle.write(f"{combo.method_name},{combo.dataset_name}\n")
        print()
        print(f"Escribi {len(pending)} combos pendientes en: {args.write_pairs}")


if __name__ == "__main__":
    main()
