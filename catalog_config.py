from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


FINAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = FINAL_DIR.parent
DEFAULT_CATALOG = FINAL_DIR / "catalog.json"
DEFAULT_RESULTS_DIR = FINAL_DIR / "results"
DEFAULT_PYTHON_CANDIDATES = (
    REPO_ROOT / "venv313" / "bin" / "python",
    REPO_ROOT / "venv" / "bin" / "python",
)


def resolve_python(python_arg: Path | None) -> Path:
    if python_arg is not None:
        return python_arg.resolve()
    for candidate in DEFAULT_PYTHON_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()
    return Path(sys.executable).resolve()


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_").lower()


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def render_template(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        return value.format(**context)
    if isinstance(value, list):
        return [render_template(item, context) for item in value]
    if isinstance(value, dict):
        return {key: render_template(item, context) for key, item in value.items()}
    return value


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_isolated_env(base_env: dict[str, str], isolation_dir: Path) -> dict[str, str]:
    tmp_dir = isolation_dir / "tmp"
    cache_dir = isolation_dir / "cache"
    home_dir = isolation_dir / "home"
    pycache_dir = isolation_dir / "pycache"
    mpl_dir = isolation_dir / "matplotlib"
    for directory in (tmp_dir, cache_dir, home_dir, pycache_dir, mpl_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = dict(base_env)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPYCACHEPREFIX"] = str(pycache_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["TMPDIR"] = str(tmp_dir)
    env["TEMP"] = str(tmp_dir)
    env["TMP"] = str(tmp_dir)
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["HOME"] = str(home_dir)
    return env


def join_command(parts: list[str]) -> str:
    return " ".join(parts)


def load_catalog(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload.get("datasets"), dict):
        raise ValueError("El catalogo debe tener una clave 'datasets' tipo objeto.")
    if not isinstance(payload.get("methods"), list):
        raise ValueError("El catalogo debe tener una clave 'methods' tipo lista.")
    return payload


def list_options(catalog: dict[str, Any]) -> None:
    print("Datasets disponibles:")
    for dataset_name, dataset_config in catalog["datasets"].items():
        description = dataset_config.get("description", "")
        print(f"  - {dataset_name}: {description}")

    print()
    print("Metodos disponibles:")
    for method in catalog["methods"]:
        datasets = ", ".join(method.get("datasets", []))
        description = method.get("description", "")
        aliases = ", ".join(method.get("aliases", []))
        print(f"  - {method['name']}: {description}")
        print(f"    datasets: {datasets}")
        if aliases:
            print(f"    aliases: {aliases}")


def select_methods(
    catalog: dict[str, Any],
    dataset_name: str,
    requested_method_names: list[str] | None,
) -> list[dict[str, Any]]:
    if dataset_name not in catalog["datasets"]:
        raise ValueError(f"Dataset desconocido: {dataset_name}")

    compatible_methods = [
        method for method in catalog["methods"]
        if dataset_name in method.get("datasets", [])
    ]
    if not compatible_methods:
        raise ValueError(f"No hay metodos compatibles con el dataset '{dataset_name}'.")

    if not requested_method_names:
        return compatible_methods

    by_name: dict[str, dict[str, Any]] = {}
    for method in compatible_methods:
        names = [method["name"], *method.get("aliases", [])]
        for name in names:
            if name in by_name and by_name[name]["name"] != method["name"]:
                raise ValueError(
                    f"Alias o nombre duplicado para el dataset '{dataset_name}': {name}"
                )
            by_name[name] = method

    missing = [name for name in requested_method_names if name not in by_name]
    if missing:
        raise ValueError(
            f"Estos metodos no son validos para el dataset '{dataset_name}': {', '.join(missing)}"
        )

    selected_methods: list[dict[str, Any]] = []
    seen_method_names: set[str] = set()
    for name in requested_method_names:
        method = by_name[name]
        canonical_name = method["name"]
        if canonical_name in seen_method_names:
            continue
        seen_method_names.add(canonical_name)
        selected_methods.append(method)
    return selected_methods
