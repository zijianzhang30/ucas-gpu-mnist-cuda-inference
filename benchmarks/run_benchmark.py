#!/usr/bin/env python3
"""Repeat the four public MNIST inference runs and preserve their provenance."""

import argparse
import csv
import hashlib
import json
import platform
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


CHECKPOINTS = {
    "lenet_96.90.bin": 9690,
    "lenet_97.07.bin": 9707,
    "deep_97.45.bin": 9745,
    "deep_99.52.bin": 9952,
}
METRICS = re.compile(
    r"model=(\w+) images=(\d+) correct=(\d+) accuracy=([\d.]+)% "
    r"kernel_ms=([\d.]+) end_to_end_ms=([\d.]+) kernel_us_per_image=([\d.]+)"
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command_output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", default="./mnist_infer")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--build-command", help="Actual build command, if known; recorded without executing it")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.batch <= 4096:
        parser.error("repeats must be positive and batch must be in [1, 4096]")
    root = Path(__file__).resolve().parents[1]
    executable = Path(args.executable).resolve()
    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "runs.json").exists():
        parser.error("output already contains runs.json; choose a new output directory")
    images = data_dir / "t10k-images-idx3-ubyte"
    labels = data_dir / "t10k-labels-idx1-ubyte"
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "source_commit": command_output(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "git_status_before_run": command_output(["git", "-C", str(root), "status", "--short"]),
        "batch_size": args.batch,
        "repeats_per_checkpoint": args.repeats,
        "run_order": "interleaved: each repetition runs all four checkpoints in listed order",
        "nvcc": command_output(["nvcc", "--version"]),
        "nvidia_smi": command_output(["nvidia-smi"]),
        "build_command": args.build_command,
        "sha256": {
            "src/mnist_infer.cu": sha256(root / "src/mnist_infer.cu"),
            "Makefile": sha256(root / "Makefile"),
            "executable": sha256(executable),
            "data/t10k-images-idx3-ubyte": sha256(images),
            "data/t10k-labels-idx1-ubyte": sha256(labels),
            **{"checkpoints/" + name: sha256(root / "checkpoints" / name) for name in CHECKPOINTS},
        },
    }
    try:
        import torch

        props = torch.cuda.get_device_properties(0)
        metadata["gpu"] = {
            "name": props.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "reported_total_memory_bytes": props.total_memory,
            "reported_multiprocessor_count": props.multi_processor_count,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    except (ImportError, RuntimeError, AssertionError) as error:
        metadata["gpu_detection_error"] = str(error)
    write_json(output / "metadata.json", metadata)
    runs = []
    try:
        for repetition in range(1, args.repeats + 1):
            for name, expected_correct in CHECKPOINTS.items():
                started = datetime.now(timezone.utc).isoformat()
                result = subprocess.run(
                    [str(executable), str(root / "checkpoints" / name), str(images), str(labels), str(args.batch)],
                    capture_output=True, text=True, timeout=120,
                )
                run = {
                    "checkpoint": name, "repetition": repetition, "started_at_utc": started,
                    "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                }
                match = METRICS.fullmatch(result.stdout.strip())
                if match:
                    run.update(zip(
                        ["model", "images", "correct", "accuracy_pct", "kernel_ms", "end_to_end_ms", "kernel_us_per_image"],
                        [match[1], int(match[2]), int(match[3]), float(match[4]), float(match[5]), float(match[6]), float(match[7])],
                    ))
                runs.append(run)
                write_json(output / "runs.json", runs)
                print(f"{repetition}/{args.repeats} {name}: {result.stdout.strip()}", flush=True)
                if result.returncode or not match:
                    raise RuntimeError(f"inference failed for {name}: {result.stderr}")
                if run["images"] != 10000 or run["correct"] != expected_correct:
                    raise RuntimeError(f"accuracy mismatch for {name}: {run['correct']}/{run['images']}")
        rows = []
        for name, correct in CHECKPOINTS.items():
            checkpoint_runs = [run for run in runs if run["checkpoint"] == name]
            row = {"checkpoint": name, "runs": len(checkpoint_runs), "images": 10000, "correct": correct, "accuracy_pct": correct / 100}
            for metric in ["kernel_ms", "end_to_end_ms"]:
                values = [run[metric] for run in checkpoint_runs]
                for statistic, value in [
                    ("median", statistics.median(values)), ("mean", statistics.mean(values)),
                    ("min", min(values)), ("max", max(values)), ("population_stddev", statistics.pstdev(values)),
                ]:
                    row[metric + "_" + statistic] = round(value, 4)
            rows.append(row)
        with (output / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        metadata["status"] = "passed"
        print(json.dumps(rows, indent=2), flush=True)
    except Exception as error:
        metadata["status"] = "failed"
        metadata["failure"] = str(error)
        raise
    finally:
        metadata["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "metadata.json", metadata)


if __name__ == "__main__":
    main()
