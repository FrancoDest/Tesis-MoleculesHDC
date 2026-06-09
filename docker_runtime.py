from __future__ import annotations

import subprocess
from pathlib import Path

from catalog_config import REPO_ROOT, join_command, render_template


def image_exists(image_name: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image_name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def build_docker_image(
    method_config: dict,
    template_context: dict[str, str],
    rebuild: bool,
    dry_run: bool,
) -> None:
    docker_config = method_config["docker"]
    image_name = render_template(docker_config["image"], template_context)

    if not rebuild and image_exists(image_name):
        if dry_run:
            print(f"[dry-run] image existente: {image_name}")
        return

    build_context = (REPO_ROOT / docker_config["build_context"]).resolve()
    dockerfile = docker_config.get("dockerfile")

    command = ["docker", "build"]
    if docker_config.get("platform"):
        command.extend(["--platform", docker_config["platform"]])
    command.extend(["-t", image_name])
    if dockerfile:
        command.extend(["-f", str((REPO_ROOT / dockerfile).resolve())])
    command.append(str(build_context))

    if dry_run:
        print(f"[dry-run] build: {join_command(command)}")
        return

    subprocess.run(command, check=True)


def build_docker_run_command(
    method_config: dict,
    template_context: dict[str, str],
) -> list[str]:
    docker_config = method_config["docker"]
    image_name = render_template(docker_config["image"], template_context)

    command = ["docker", "run", "--rm"]
    if docker_config.get("platform"):
        command.extend(["--platform", docker_config["platform"]])
    command.extend(["-e", "PYTHONUNBUFFERED=1", "-e", "PYTHONIOENCODING=utf-8"])

    for mount in docker_config.get("mounts", []):
        source = render_template(mount["source"], template_context)
        target = render_template(mount["target"], template_context)
        command.extend(["-v", f"{source}:{target}"])

    workdir = render_template(docker_config["workdir"], template_context)
    command.extend(["-w", workdir, image_name])

    run_args = render_template(docker_config.get("run_args", []), template_context)
    command.extend(run_args)
    return command
