#!/usr/bin/env python

import argparse
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from transformers import AutoModel, AutoModelForCausalLM


def _read_rss_bytes() -> int:
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(usage)
    return int(usage) * 1024


class PeakRssTracker:
    def __init__(self, interval: float = 0.02):
        self.interval = interval
        self.peak_rss = _read_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, _read_rss_bytes())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, _read_rss_bytes())


def _format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def _init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    return local_rank, rank, world_size


def _pick_model_class(task: str):
    if task == "causal-lm":
        return AutoModelForCausalLM
    return AutoModel


def _worker(args: argparse.Namespace) -> None:
    local_rank, rank, world_size = _init_dist()
    hostname = socket.gethostname()

    os.environ["ACCELERATE_USE_FSDP"] = "True"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True" if args.cpu_ram_efficient_loading else "False"
    # Mirrors Trainer/Accelerate usage: CPU RAM efficient loading is only valid when FSDP
    # syncs module states from rank 0 after non-main ranks start with empty placeholders.
    if args.cpu_ram_efficient_loading and not args.sync_module_states:
        raise ValueError("`--sync-module-states` must be enabled when `--cpu-ram-efficient-loading` is set.")

    model_cls = _pick_model_class(args.task)
    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)

    start_rss = _read_rss_bytes()
    start_time = time.perf_counter()
    with PeakRssTracker() as tracker:
        model = model_cls.from_pretrained(
            args.model_name_or_path,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            torch_dtype=dtype if dtype is not None else "auto",
            trust_remote_code=args.trust_remote_code,
        )
        if args.wrap_fsdp:
            model = FSDP(model, sync_module_states=args.sync_module_states)
        load_done_time = time.perf_counter()
        after_load_rss = _read_rss_bytes()

        num_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

        # Keep the model alive until after all ranks report, otherwise RSS can drop too early.
        dist.barrier()

    metrics = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": hostname,
        "cpu_ram_efficient_loading": args.cpu_ram_efficient_loading,
        "sync_module_states": args.sync_module_states,
        "wrap_fsdp": args.wrap_fsdp,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "model_name_or_path": args.model_name_or_path,
        "load_time_s": round(load_done_time - start_time, 3),
        "rss_start_gib": _format_gib(start_rss),
        "rss_after_load_gib": _format_gib(after_load_rss),
        "rss_peak_gib": _format_gib(tracker.peak_rss),
        "rss_delta_gib": _format_gib(after_load_rss - start_rss),
        "rss_peak_delta_gib": _format_gib(tracker.peak_rss - start_rss),
        "num_params": num_params,
        "trainable_params": trainable_params,
    }

    gathered: list[dict | None] = [None for _ in range(world_size)] if rank == 0 else []
    dist.gather_object(metrics, gathered, dst=0)

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mode = "enabled" if args.cpu_ram_efficient_loading else "disabled"
        output_file = output_dir / f"fsdp_cpu_ram_efficient_loading_{mode}.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(gathered, handle, indent=2, sort_keys=True)

        peak = max(item["rss_peak_gib"] for item in gathered)
        delta = max(item["rss_peak_delta_gib"] for item in gathered)
        print(f"[{mode}] wrote {output_file}")
        print(f"[{mode}] max rank peak RSS: {peak} GiB")
        print(f"[{mode}] max rank peak RSS delta: {delta} GiB")

    dist.barrier()
    dist.destroy_process_group()


def _launcher(args: argparse.Namespace) -> None:
    script_path = Path(__file__).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(args.nproc_per_node),
        "--nnodes",
        "1",
        str(script_path),
        "--worker",
        "--model-name-or-path",
        args.model_name_or_path,
        "--task",
        args.task,
        "--output-dir",
        str(output_dir),
        "--dtype",
        args.dtype,
        "--sync-module-states",
    ]
    if args.low_cpu_mem_usage:
        base_cmd.append("--low-cpu-mem-usage")
    if args.trust_remote_code:
        base_cmd.append("--trust-remote-code")
    if args.wrap_fsdp:
        base_cmd.append("--wrap-fsdp")

    for enabled in (False, True):
        cmd = [*base_cmd, "--cpu-ram-efficient-loading"] if enabled else base_cmd
        label = "enabled" if enabled else "disabled"
        print(f"Launching benchmark with FSDP_CPU_RAM_EFFICIENT_LOADING={label}")
        subprocess.run(cmd, check=True)

    disabled_path = output_dir / "fsdp_cpu_ram_efficient_loading_disabled.json"
    enabled_path = output_dir / "fsdp_cpu_ram_efficient_loading_enabled.json"
    with disabled_path.open("r", encoding="utf-8") as handle:
        disabled = json.load(handle)
    with enabled_path.open("r", encoding="utf-8") as handle:
        enabled = json.load(handle)

    def _max_metric(rows: list[dict], key: str) -> float:
        return max(row[key] for row in rows)

    summary = {
        "model_name_or_path": args.model_name_or_path,
        "world_size": args.nproc_per_node,
        "disabled_max_peak_rss_gib": _max_metric(disabled, "rss_peak_gib"),
        "enabled_max_peak_rss_gib": _max_metric(enabled, "rss_peak_gib"),
        "disabled_max_peak_delta_gib": _max_metric(disabled, "rss_peak_delta_gib"),
        "enabled_max_peak_delta_gib": _max_metric(enabled, "rss_peak_delta_gib"),
        "disabled_max_load_time_s": _max_metric(disabled, "load_time_s"),
        "enabled_max_load_time_s": _max_metric(enabled, "load_time_s"),
    }
    summary["peak_delta_saved_gib"] = round(
        summary["disabled_max_peak_delta_gib"] - summary["enabled_max_peak_delta_gib"], 3
    )

    summary_path = output_dir / "fsdp_cpu_ram_efficient_loading_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"Wrote summary to {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare CPU memory usage of Transformers weight loading with and without FSDP_CPU_RAM_EFFICIENT_LOADING."
    )
    parser.add_argument("--worker", action="store_true", help="Internal flag used by the launcher.")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--task", choices=("base", "causal-lm"), default="causal-lm")
    parser.add_argument("--nproc-per-node", type=int, default=torch.cuda.device_count() or 1)
    parser.add_argument("--output-dir", default="tmp/fsdp_cpu_ram_efficient_loading_bench")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cpu-ram-efficient-loading", action="store_true")
    parser.add_argument("--sync-module-states", action="store_true")
    parser.add_argument("--wrap-fsdp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        _worker(args)
    else:
        _launcher(args)


if __name__ == "__main__":
    main()
