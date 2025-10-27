import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence

import pandas as pd
from utils.config import (
    get_config_path,
    load_config,
    resolve_path as resolve_config_path,
    reset_config_cache,
)

try:
    import torch  # type: ignore
    import torch.distributed as dist  # type: ignore
except ImportError:
    torch = None  # type: ignore
    dist = None  # type: ignore

TASK_NAMES: Sequence[str] = (
    "data_analysis",
    "vision_memory",
    "llm_memory",
    "vision_throughput",
    "llm_throughput_full",
    "llm_throughput_skip_attn",
)

BASE_DIR = Path(__file__).resolve().parent


@dataclass
class ProfilingTask:
    name: str
    script: str
    arguments: List[str]
    run_once: bool = False
    env: Optional[Dict[str, str]] = None

    def build_command(self, scripts_dir: Path, profiler_script: Path, config_path: Path) -> List[str]:
        script_path = scripts_dir / self.script
        if not script_path.exists():
            raise FileNotFoundError(f"Profiling script not found: {script_path}")
        return ["bash", str(script_path), str(profiler_script), str(config_path), *self.arguments]


@dataclass
class DistContext:
    distributed: bool
    initialized_here: bool
    rank: int
    world_size: int
    backend: Optional[str] = None


def setup_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def resolve_paths(settings: SimpleNamespace) -> tuple[Path, Path, Path]:
    scripts_dir = (
        resolve_config_path(settings.scripts_dir) if settings.scripts_dir else (BASE_DIR / "scripts").resolve()
    )
    profiler_script = (
        resolve_config_path(settings.profiler_script)
        if settings.profiler_script
        else (BASE_DIR / "prof_engine.py").resolve()
    )
    results_csv = (
        resolve_config_path(settings.results_csv)
        if settings.results_csv
        else (BASE_DIR / "profile_results.csv").resolve()
    )
    return scripts_dir, profiler_script, results_csv


def build_catalog(settings: SimpleNamespace) -> List[ProfilingTask]:
    catalog = {
        "data_analysis": ProfilingTask(
            "data_analysis",
            "run_dataset_analysis.sh",
            [],
            run_once=True,
            env={"PROFILE_MODE": "data"},
        ),
        "vision_memory": ProfilingTask(
            "vision_memory",
            "run_mem_vision.sh",
            [],
            env={"PROFILE_MODE": "mem"},
        ),
        "llm_memory": ProfilingTask(
            "llm_memory",
            "run_mem_llm.sh",
            [],
            env={"PROFILE_MODE": "mem"},
        ),
        "vision_throughput": ProfilingTask(
            "vision_throughput",
            "run_thr_vision.sh",
            [],
            env={"PROFILE_MODE": "thr"},
        ),
        "llm_throughput_full": ProfilingTask(
            "llm_throughput_full",
            "run_thr_llm_full.sh",
            [],
            env={"PROFILE_MODE": "thr"},
        ),
        "llm_throughput_skip_attn": ProfilingTask(
            "llm_throughput_skip_attn",
            "run_thr_llm_skip_attn.sh",
            [],
            env={"PROFILE_MODE": "thr", "SKIP_ATTN": "1"},
        ),
    }
    task_order = settings.tasks if settings.tasks else TASK_NAMES
    return [catalog[name] for name in task_order]


def validate_environment(scripts_dir: Path, profiler_script: Path, tasks: Sequence[ProfilingTask]) -> None:
    if not scripts_dir.is_dir():
        raise NotADirectoryError(f"Profiling scripts directory does not exist: {scripts_dir}")
    if not profiler_script.exists():
        raise FileNotFoundError(f"Profiler script not found: {profiler_script}")
    missing_scripts = [task.script for task in tasks if not (scripts_dir / task.script).exists()]
    if missing_scripts:
        raise FileNotFoundError(
            f"Missing profiling shell scripts: {', '.join(sorted(missing_scripts))} (looked under {scripts_dir})"
        )


def setup_distributed(backend: Optional[str]) -> DistContext:
    if dist is None or not hasattr(dist, "is_available") or not dist.is_available():
        return DistContext(distributed=False, initialized_here=False, rank=0, world_size=1, backend=None)

    world_size_env = os.environ.get("WORLD_SIZE")
    if world_size_env is None:
        return DistContext(distributed=False, initialized_here=False, rank=0, world_size=1, backend=None)

    backend_to_use = backend
    if backend_to_use is None:
        backend_to_use = "nccl" if torch is not None and torch.cuda.is_available() else "gloo"

    initialized_here = False
    if not dist.is_initialized():
        dist.init_process_group(backend=backend_to_use)
        initialized_here = True

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    distributed = world_size > 1
    return DistContext(
        distributed=distributed,
        initialized_here=initialized_here,
        rank=rank,
        world_size=world_size,
        backend=backend_to_use,
    )


def assign_tasks(tasks: Sequence[ProfilingTask], ctx: DistContext) -> List[ProfilingTask]:
    if not ctx.distributed:
        return list(tasks)

    shared_tasks = [task for task in tasks if not task.run_once]
    run_once_tasks = [task for task in tasks if task.run_once]

    assigned: List[ProfilingTask] = []
    if ctx.rank == 0:
        assigned.extend(run_once_tasks)

    for idx, task in enumerate(shared_tasks):
        if idx % ctx.world_size == ctx.rank:
            assigned.append(task)

    return assigned


def execute_tasks(
    tasks: Sequence[ProfilingTask],
    ctx: DistContext,
    scripts_dir: Path,
    profiler_script: Path,
    config_path: Path,
    stop_on_failure: bool,
    base_env: Dict[str, str],
) -> tuple[List[dict], Optional[subprocess.CalledProcessError]]:
    results: List[dict] = []
    failure: Optional[subprocess.CalledProcessError] = None

    for task in tasks:
        command = task.build_command(scripts_dir, profiler_script, config_path)
        command_str = " ".join(shlex.quote(part) for part in command)
        logging.info("Rank %s executing task '%s': %s", ctx.rank, task.name, command_str)
        started_at = time.time()
        status = "success"
        return_code = 0
        error_message = ""

        env = base_env.copy()
        if task.env:
            env.update(task.env)

        try:
            subprocess.run(command, check=True, env=env)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            return_code = exc.returncode
            error_message = str(exc)
            failure = exc
            logging.error(
                "Rank %s task '%s' failed with return code %s",
                ctx.rank,
                task.name,
                return_code,
            )
            if stop_on_failure:
                logging.warning("Rank %s stopping after failure (stop-on-failure enabled).", ctx.rank)

        finished_at = time.time()
        results.append(
            {
                "task": task.name,
                "status": status,
                "duration_sec": finished_at - started_at,
                "rank": ctx.rank,
                "return_code": return_code,
                "command": command_str,
                "error": error_message,
            }
        )

        if failure is not None and stop_on_failure:
            break

    if not tasks:
        logging.info("Rank %s: no profiling tasks assigned.", ctx.rank)

    return results, failure


def gather_results(local_results: List[dict], ctx: DistContext) -> List[dict]:
    if ctx.distributed and dist is not None:
        dist.barrier()
        gather_list = [None] * ctx.world_size if ctx.rank == 0 else None
        dist.gather_object(local_results, gather_list, dst=0)
        if ctx.rank == 0 and gather_list is not None:
            aggregated: List[dict] = []
            for worker_results in gather_list:
                if worker_results:
                    aggregated.extend(worker_results)
            return aggregated
        return []
    return list(local_results)


def write_results(
    all_results: List[dict],
    tasks: Sequence[ProfilingTask],
    settings: SimpleNamespace,
    results_csv: Path,
) -> None:
    if not all_results:
        logging.warning("No profiling results to record; skipping CSV update.")
        return

    order_index = {task.name: index for index, task in enumerate(tasks)}
    all_results.sort(key=lambda item: (order_index.get(item["task"], len(order_index)), item["rank"]))

    for row in all_results:
        row.update(
            {
                "mllm_model": settings.mllm_model_name,
                "vision_model": settings.vision_model_name,
                "vision_model_size": settings.vision_model_size,
                "llm_model": settings.llm_model_name,
                "llm_model_size": settings.llm_model_size,
            }
        )

    df = pd.DataFrame(all_results)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    if results_csv.exists():
        existing_df = pd.read_csv(results_csv)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_csv(results_csv, index=False)
    logging.info("Recorded %s profiling result(s) to %s", len(all_results), results_csv)


def main() -> None:
    default_config_path = get_config_path()
    os.environ.setdefault("DFLOP_CONFIG", str(default_config_path))
    reset_config_cache()
    config = load_config()
    config_path = Path(os.environ["DFLOP_CONFIG"]).resolve()

    profiling_cfg = config.get("profiling", {})
    models_cfg = config.get("models", {})
    vision_cfg = models_cfg.get("vision", {})
    llm_cfg = models_cfg.get("llm", {})
    paths_cfg = config.get("paths", {})

    settings = SimpleNamespace(
        mllm_model_name=models_cfg.get("mllm"),
        vision_model_name=vision_cfg.get("name"),
        vision_model_size=vision_cfg.get("size"),
        llm_model_name=llm_cfg.get("name"),
        llm_model_size=llm_cfg.get("size"),
        tasks=profiling_cfg.get("tasks"),
        scripts_dir=paths_cfg.get("profiling_scripts_dir"),
        profiler_script=paths_cfg.get("profiler_script"),
        results_csv=paths_cfg.get("profile_results_csv"),
        dist_backend=profiling_cfg.get("dist_backend"),
        stop_on_failure=profiling_cfg.get("stop_on_failure", False),
        log_level=profiling_cfg.get("log_level", "INFO"),
    )

    required_fields = [
        "mllm_model_name",
        "vision_model_name",
        "vision_model_size",
        "llm_model_name",
        "llm_model_size",
    ]
    missing = [field for field in required_fields if getattr(settings, field) is None]
    if missing:
        raise ValueError(f"Missing profiling configuration values: {', '.join(missing)}")

    setup_logging(settings.log_level)

    scripts_dir, profiler_script, results_csv = resolve_paths(settings)
    tasks = build_catalog(settings)
    if not tasks:
        logging.warning("No tasks selected; nothing to do.")
        return

    validate_environment(scripts_dir, profiler_script, tasks)

    ctx = setup_distributed(settings.dist_backend)
    logging.info(
        "Distributed context - rank: %s, world_size: %s, backend: %s, distributed: %s",
        ctx.rank,
        ctx.world_size,
        ctx.backend,
        ctx.distributed,
    )

    assigned_tasks = assign_tasks(tasks, ctx)
    logging.info(
        "Rank %s assigned tasks: %s",
        ctx.rank,
        ", ".join(task.name for task in assigned_tasks) or "(none)",
    )

    base_env = os.environ.copy()
    base_env.setdefault("DFLOP_CONFIG", str(config_path))

    local_results, failure = execute_tasks(
        assigned_tasks,
        ctx,
        scripts_dir,
        profiler_script,
        config_path,
        stop_on_failure=settings.stop_on_failure,
        base_env=base_env,
    )

    aggregated_results = gather_results(local_results, ctx)

    if ctx.initialized_here and dist is not None and dist.is_initialized():
        dist.destroy_process_group()

    if ctx.rank == 0:
        write_results(aggregated_results, tasks, settings, results_csv)

    if failure is not None and settings.stop_on_failure:
        sys.exit(failure.returncode)


if __name__ == "__main__":
    main()
