#!/usr/bin/env python

import json
import os
import resource
import socket
import sys
import threading
import time
from pathlib import Path

import torch

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments


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
    def __init__(self, interval_s: float = 0.02):
        self.interval_s = interval_s
        self.peak_rss = _read_rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss = max(self.peak_rss, _read_rss_bytes())
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._thread.join()
        self.peak_rss = max(self.peak_rss, _read_rss_bytes())


def _format_gib(num_bytes: int) -> float:
    return round(num_bytes / (1024**3), 3)


def _pop_arg(name: str, default=None):
    if name not in sys.argv:
        return default
    idx = sys.argv.index(name)
    value = sys.argv[idx + 1]
    sys.argv.pop(idx)
    sys.argv.pop(idx)
    return value


def _pop_flag(name: str) -> bool:
    if name not in sys.argv:
        return False
    sys.argv.remove(name)
    return True


def main() -> None:
    model_name_or_path = _pop_arg("--model_name_or_path")
    model_dtype = _pop_arg("--model_dtype", "bfloat16")
    metrics_output = _pop_arg("--metrics_output", "tmp/accelerate_fsdp2_cpu_ram_peak.json")
    low_cpu_mem_usage = _pop_flag("--low_cpu_mem_usage")
    trust_remote_code = _pop_flag("--trust_remote_code")

    if model_name_or_path is None:
        raise ValueError("`--model_name_or_path` is required.")

    parser = HfArgumentParser((TrainingArguments,))
    (training_args,) = parser.parse_args_into_dataclasses()

    dtype = None if model_dtype == "auto" else getattr(torch, model_dtype)
    rank = training_args.process_index
    world_size = training_args.world_size
    local_rank = training_args.local_process_index
    hostname = socket.gethostname()

    start_rss = _read_rss_bytes()
    start_time = time.perf_counter()

    with PeakRssTracker() as tracker:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype if dtype is not None else "auto",
            low_cpu_mem_usage=low_cpu_mem_usage,
            trust_remote_code=trust_remote_code,
        )
        after_load_rss = _read_rss_bytes()
        trainer = Trainer(model=model, args=training_args)
        after_trainer_rss = _read_rss_bytes()
        load_time_s = round(time.perf_counter() - start_time, 3)

        num_params = sum(param.numel() for param in model.parameters())
        trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    metrics = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "hostname": hostname,
        "model_name_or_path": model_name_or_path,
        "fsdp": training_args.fsdp,
        "fsdp_config": training_args.fsdp_config,
        "load_time_s": load_time_s,
        "rss_start_gib": _format_gib(start_rss),
        "rss_after_load_gib": _format_gib(after_load_rss),
        "rss_after_trainer_gib": _format_gib(after_trainer_rss),
        "rss_peak_gib": _format_gib(tracker.peak_rss),
        "rss_delta_gib": _format_gib(after_load_rss - start_rss),
        "rss_peak_delta_gib": _format_gib(tracker.peak_rss - start_rss),
        "num_params": num_params,
        "trainable_params": trainable_params,
        "low_cpu_mem_usage": low_cpu_mem_usage,
        "env_ACCELERATE_USE_FSDP": os.environ.get("ACCELERATE_USE_FSDP"),
        "env_FSDP_CPU_RAM_EFFICIENT_LOADING": os.environ.get("FSDP_CPU_RAM_EFFICIENT_LOADING"),
    }

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.gather_object(metrics, gathered, dst=0)
    else:
        gathered = [metrics]

    if rank == 0:
        output_path = Path(metrics_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(gathered, handle, indent=2, sort_keys=True)
        print(f"Wrote metrics to {output_path}")
        print(json.dumps(gathered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
