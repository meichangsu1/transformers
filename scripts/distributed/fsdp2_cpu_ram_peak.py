#!/usr/bin/env python

import argparse
import json
import os
import resource
import socket
import sys
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard

from transformers import AutoModelForCausalLM


def read_rss_bytes() -> int:
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
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak_rss = read_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, read_rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, read_rss_bytes())


def format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def init_dist() -> tuple[int, int, int]:
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend)
    return local_rank, rank, world_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CPU RSS peak for HF model loading with FSDP2, toggling FSDP_CPU_RAM_EFFICIENT_LOADING."
    )
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--output-dir", default="tmp/fsdp2_cpu_ram_peak")
    parser.add_argument("--dtype", default="bfloat16", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--enable-cpu-ram-efficient-loading", action="store_true")
    parser.add_argument("--low-cpu-mem-usage", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank, rank, world_size = init_dist()

    os.environ["ACCELERATE_USE_FSDP"] = "True"
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "True" if args.enable_cpu_ram_efficient_loading else "False"

    dtype = None if args.dtype == "auto" else getattr(torch, args.dtype)
    hostname = socket.gethostname()
    start_rss = read_rss_bytes()
    start_time = time.perf_counter()

    with PeakRssTracker() as tracker:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype if dtype is not None else "auto",
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
        after_load_rss = read_rss_bytes()
        load_time_s = round(time.perf_counter() - start_time, 3)

        # Use FSDP2 composable API after loading so the run matches an FSDP2 training setup.
        for submodule in model.modules():
            if submodule is model:
                continue
            if len(list(submodule.children())) == 0:
                continue
            try:
                fully_shard(submodule)
            except Exception:
                continue
        fully_shard(model)
        after_fsdp2_rss = read_rss_bytes()

        num_params = sum(param.numel() for param in model.parameters())
        dist.barrier()

    metrics = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": hostname,
        "cpu_ram_efficient_loading": args.enable_cpu_ram_efficient_loading,
        "low_cpu_mem_usage": args.low_cpu_mem_usage,
        "model_name_or_path": args.model_name_or_path,
        "load_time_s": load_time_s,
        "rss_start_gib": format_gib(start_rss),
        "rss_after_load_gib": format_gib(after_load_rss),
        "rss_after_fsdp2_gib": format_gib(after_fsdp2_rss),
        "rss_peak_gib": format_gib(tracker.peak_rss),
        "rss_delta_gib": format_gib(after_load_rss - start_rss),
        "rss_peak_delta_gib": format_gib(tracker.peak_rss - start_rss),
        "num_params": num_params,
    }

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(metrics, gathered, dst=0)

    if rank == 0:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        mode = "enabled" if args.enable_cpu_ram_efficient_loading else "disabled"
        out_file = output_dir / f"fsdp2_cpu_ram_peak_{mode}.json"
        with out_file.open("w", encoding="utf-8") as handle:
            json.dump(gathered, handle, indent=2, sort_keys=True)
        print(f"Wrote metrics to {out_file}")
        print(json.dumps(gathered, indent=2, sort_keys=True))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
# PYTHONPATH=src accelerate launch \
#   --config_file tests/trainer/distributed/accelerate_configs/fsdp2.yaml \
#   --num_processes 4 \
#   scripts/distributed/accelerate_fsdp2_cpu_ram_peak.py \
#   --model_name_or_path /nas/disk1/Qwen3-8B \
#   --model_dtype bfloat16 \
#   --low_cpu_mem_usage \
#   --output_dir tmp/bench_disabled \
#   --fsdp_cpu_ram_efficient_loading false
# PYTHONPATH=src accelerate launch \
#   --config_file tests/trainer/distributed/accelerate_configs/fsdp2.yaml \
#   --num_processes 4 \
#   scripts/distributed/accelerate_fsdp2_cpu_ram_peak.py \
#   --model_name_or_path /nas/disk1/Qwen3-8B \
#   --model_dtype bfloat16 \
#   --low_cpu_mem_usage \
#   --output_dir tmp/bench_enabled \
#   --fsdp_cpu_ram_efficient_loading true
