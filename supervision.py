#!/usr/bin/env python3
"""Supervise two ComfyUI processes and image_generation_server on RunPod.

Starts ComfyUI from /workspace/ComfyUI as:

    CUDA_VISIBLE_DEVICES=0 python main.py --port 8181 --listen 0.0.0.0
    CUDA_VISIBLE_DEVICES=1 python main.py --port 8182 --listen 0.0.0.0

Starts the backend server from /workspace as:

    python image_generation_server.py
"""

from __future__ import annotations

import argparse
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


STOP_EVENT = threading.Event()
MANAGED_PROCESSES: list[subprocess.Popen[bytes]] = []
PROCESS_LOCK = threading.Lock()


@dataclass(frozen=True)
class ComfyInstance:
    gpu: int
    port: int


@dataclass(frozen=True)
class SupervisorConfig:
    comfy_instances: tuple[ComfyInstance, ...]
    comfy_dir: Path
    workspace_dir: Path
    log_dir: Path
    restart_delay: int
    health_interval: int
    health_grace: int
    health_failures: int
    health_timeout: int
    comfy_extra_args: list[str]
    start_image_server: bool


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer, got {value!r}") from exc


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str, supervisor_log: Path | None = None) -> None:
    line = f"{timestamp()} [supervisor] {message}"
    print(line, flush=True)
    if supervisor_log is not None:
        supervisor_log.parent.mkdir(parents=True, exist_ok=True)
        with supervisor_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def health_url(port: int) -> str:
    return f"http://127.0.0.1:{port}/system_stats"


def is_healthy(port: int, timeout: int) -> bool:
    try:
        with urllib.request.urlopen(health_url(port), timeout=timeout) as response:
            return 200 <= response.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def describe_exit(return_code: int | None) -> str:
    if return_code is None:
        return "still running"
    if return_code in (-signal.SIGKILL, 137):
        return f"exited with code {return_code}, likely OOM/SIGKILL"
    if return_code < 0:
        return f"exited from signal {-return_code}"
    return f"exited with code {return_code}"


def register_process(process: subprocess.Popen[bytes]) -> None:
    with PROCESS_LOCK:
        MANAGED_PROCESSES.append(process)


def unregister_process(process: subprocess.Popen[bytes]) -> None:
    with PROCESS_LOCK:
        if process in MANAGED_PROCESSES:
            MANAGED_PROCESSES.remove(process)


def stop_process(
    process: subprocess.Popen[bytes],
    supervisor_log: Path,
    name: str,
) -> None:
    if process.poll() is not None:
        return

    log(f"Terminating {name} PID {process.pid}", supervisor_log)
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        log(f"Force killing {name} PID {process.pid}", supervisor_log)
        process.kill()
        process.wait(timeout=10)


def start_comfyui(
    config: SupervisorConfig,
    instance: ComfyInstance,
    comfy_log: Path,
    supervisor_log: Path,
) -> tuple[subprocess.Popen[bytes], object]:
    config.comfy_dir.mkdir(parents=True, exist_ok=True)
    comfy_log.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "main.py",
        "--port",
        str(instance.port),
        "--listen",
        "0.0.0.0",
        *config.comfy_extra_args,
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(instance.gpu)

    log(
        f"Starting ComfyUI GPU {instance.gpu} on port {instance.port}: "
        + f"CUDA_VISIBLE_DEVICES={instance.gpu} "
        + " ".join(shlex.quote(part) for part in command),
        supervisor_log,
    )

    log_handle = comfy_log.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=config.comfy_dir,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    register_process(process)
    log(f"ComfyUI PID {process.pid} -> {comfy_log}", supervisor_log)
    return process, log_handle


def start_image_server(
    config: SupervisorConfig,
    server_log: Path,
    supervisor_log: Path,
) -> tuple[subprocess.Popen[bytes], object]:
    config.workspace_dir.mkdir(parents=True, exist_ok=True)
    server_log.parent.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, "image_generation_server.py"]

    log(
        "Starting image generation server: "
        + " ".join(shlex.quote(part) for part in command),
        supervisor_log,
    )

    log_handle = server_log.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=config.workspace_dir,
        env=os.environ.copy(),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )

    register_process(process)
    log(f"Image generation server PID {process.pid} -> {server_log}", supervisor_log)
    return process, log_handle


def supervise_image_server(config: SupervisorConfig) -> None:
    server_log = config.log_dir / "image_generation_server.log"
    supervisor_log = config.log_dir / "image_generation_server_supervisor.log"

    process, log_handle = start_image_server(config, server_log, supervisor_log)

    while not STOP_EVENT.is_set():
        time.sleep(config.health_interval)

        return_code = process.poll()
        if return_code is None:
            continue

        log(
            f"Image generation server PID {process.pid} {describe_exit(return_code)}.",
            supervisor_log,
        )
        log_handle.close()
        unregister_process(process)

        if STOP_EVENT.is_set():
            break

        time.sleep(config.restart_delay)
        process, log_handle = start_image_server(config, server_log, supervisor_log)

    stop_process(process, supervisor_log, "image generation server")
    log_handle.close()
    unregister_process(process)


def supervise_comfy_instance(config: SupervisorConfig, instance: ComfyInstance) -> None:
    comfy_log = config.log_dir / f"comfyui_gpu{instance.gpu}_port{instance.port}.log"
    supervisor_log = (
        config.log_dir / f"comfyui_gpu{instance.gpu}_port{instance.port}_supervisor.log"
    )

    process: subprocess.Popen[bytes] | None = None
    log_handle: object | None = None
    managed_started_at = 0.0
    failures = 0

    if is_healthy(instance.port, config.health_timeout):
        log(
            f"Port {instance.port} is already healthy. Monitoring existing ComfyUI; "
            "a managed replacement will start if it goes down.",
            supervisor_log,
        )
    else:
        process, log_handle = start_comfyui(
            config, instance, comfy_log, supervisor_log
        )
        managed_started_at = time.monotonic()

    while not STOP_EVENT.is_set():
        time.sleep(config.health_interval)

        if process is not None:
            return_code = process.poll()
            if return_code is not None:
                log(
                    f"ComfyUI PID {process.pid} {describe_exit(return_code)}.",
                    supervisor_log,
                )
                if log_handle is not None:
                    log_handle.close()
                unregister_process(process)

                if STOP_EVENT.is_set():
                    break

                time.sleep(config.restart_delay)
                process, log_handle = start_comfyui(
                    config, instance, comfy_log, supervisor_log
                )
                managed_started_at = time.monotonic()
                failures = 0
                continue

            if is_healthy(instance.port, config.health_timeout):
                failures = 0
                continue

            if time.monotonic() - managed_started_at < config.health_grace:
                continue

            failures += 1
            log(
                f"Health check failed for ComfyUI PID {process.pid} "
                f"on port {instance.port} ({failures}/{config.health_failures})",
                supervisor_log,
            )

            if failures >= config.health_failures:
                stop_process(process, supervisor_log, "ComfyUI")
                if log_handle is not None:
                    log_handle.close()
                unregister_process(process)
                process = None
                log_handle = None
                failures = 0

            continue

        if is_healthy(instance.port, config.health_timeout):
            failures = 0
            continue

        failures += 1
        log(
            f"Existing ComfyUI on port {instance.port} is unavailable "
            f"({failures}/{config.health_failures})",
            supervisor_log,
        )

        if failures < config.health_failures:
            continue

        process, log_handle = start_comfyui(
            config, instance, comfy_log, supervisor_log
        )
        managed_started_at = time.monotonic()
        failures = 0

    if process is not None:
        stop_process(process, supervisor_log, "ComfyUI")
        if log_handle is not None:
            log_handle.close()
        unregister_process(process)


def parse_args() -> SupervisorConfig:
    parser = argparse.ArgumentParser(
        description="Monitor and restart two ComfyUI instances plus image_generation_server."
    )
    parser.add_argument(
        "--comfy-dir",
        type=Path,
        default=Path(os.environ.get("COMFYUI_DIR", "/workspace/ComfyUI")),
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(os.environ.get("WORKSPACE_DIR", "/workspace")),
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(os.environ.get("SUPERVISION_LOG_DIR", "/workspace")),
    )
    parser.add_argument(
        "--restart-delay",
        type=int,
        default=env_int("SUPERVISION_RESTART_DELAY", 10),
    )
    parser.add_argument(
        "--health-interval",
        type=int,
        default=env_int("SUPERVISION_HEALTH_INTERVAL", 30),
    )
    parser.add_argument(
        "--health-grace",
        type=int,
        default=env_int("SUPERVISION_HEALTH_GRACE", 180),
    )
    parser.add_argument(
        "--health-failures",
        type=int,
        default=env_int("SUPERVISION_HEALTH_FAILURES", 5),
    )
    parser.add_argument(
        "--health-timeout",
        type=int,
        default=env_int("SUPERVISION_HEALTH_TIMEOUT", 5),
    )
    parser.add_argument(
        "--comfy-extra-args",
        default=os.environ.get("COMFYUI_EXTRA_ARGS", ""),
        help='Extra ComfyUI args, for example: "--lowvram --reserve-vram 1"',
    )
    parser.add_argument(
        "--no-image-server",
        action="store_true",
        default=not env_bool("START_IMAGE_SERVER", True),
        help="Do not start image_generation_server.py.",
    )

    args = parser.parse_args()

    return SupervisorConfig(
        comfy_instances=(
            ComfyInstance(gpu=0, port=8181),
            ComfyInstance(gpu=1, port=8182),
        ),
        comfy_dir=args.comfy_dir,
        workspace_dir=args.workspace_dir,
        log_dir=args.log_dir,
        restart_delay=max(1, args.restart_delay),
        health_interval=max(1, args.health_interval),
        health_grace=max(0, args.health_grace),
        health_failures=max(1, args.health_failures),
        health_timeout=max(1, args.health_timeout),
        comfy_extra_args=shlex.split(args.comfy_extra_args),
        start_image_server=not args.no_image_server,
    )


def handle_shutdown(signum: int, _frame: object) -> None:
    log(f"Received signal {signum}; stopping managed processes.")
    STOP_EVENT.set()

    with PROCESS_LOCK:
        processes = list(MANAGED_PROCESSES)

    for process in processes:
        if process.poll() is None:
            process.terminate()


def main() -> int:
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    config = parse_args()
    comfy_summary = ", ".join(
        f"GPU {instance.gpu} -> port {instance.port}"
        for instance in config.comfy_instances
    )

    log(f"Supervising ComfyUI instances: {comfy_summary}; logs in {config.log_dir}")

    threads: list[threading.Thread] = []

    for instance in config.comfy_instances:
        threads.append(
            threading.Thread(
                target=supervise_comfy_instance,
                args=(config, instance),
                daemon=False,
                name=f"comfyui-gpu-{instance.gpu}-port-{instance.port}",
            )
        )

    if config.start_image_server:
        threads.append(
            threading.Thread(
                target=supervise_image_server,
                args=(config,),
                daemon=False,
                name="image-generation-server",
            )
        )

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
