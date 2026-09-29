#!/usr/bin/env python3
"""
Runtime benchmark for SGWN-Encoder 2D/3D hologram synthesis.

The benchmark is intentionally separate from evaluation scripts so timing is
not polluted by disk I/O, reconstruction, or PSNR/SSIM calculation.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from Optics import Optics
from hyperparams import Hyperparams
from checkpoint_utils import load_phase_model


DEFAULT_MODEL_PATH = Path("checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/benchmark_sgwn_encoder_2d3d")
CHANNEL_NAMES = {0: "B", 1: "G", 2: "R"}
TASKS = {
    "2d_single": {"layers": 1, "channels": [1]},
    "2d_rgb": {"layers": 1, "channels": [0, 1, 2]},
    "3d8_single": {"layers": 8, "channels": [1]},
    "3d8_rgb": {"layers": 8, "channels": [0, 1, 2]},
    "3d64_single": {"layers": 64, "channels": [1]},
    "3d64_rgb": {"layers": 64, "channels": [0, 1, 2]},
}


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def select_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    Hyperparams.device = device
    Hyperparams.SAVE = False
    Hyperparams.MUL_SAVE = False
    Hyperparams.TIME = False
    return device


def load_model(model_path, device):
    return load_phase_model(model_path, device=device)


def make_optics(channel, layer_num, args):
    return Optics(
        channel=channel,
        first_z=args.z,
        delta_z=args.delta_z,
        layer_num=layer_num,
        factor=args.factor,
        grating_type=args.grating_type,
        LCoS_res_h=args.height,
        LCoS_res_w=args.width,
        LCoS_pitch=args.pitch,
        linear_conv=not args.no_linear_conv,
        band_limit=not args.no_band_limit,
    )


def load_image_channels(args, device, channels):
    if args.input_image is None:
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed)
        intensities = torch.rand(
            (1, 3, args.height, args.width),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return {channel: torch.sqrt(intensities[:, channel:channel + 1]) for channel in channels}

    image_path = resolve_path(args.input_image)
    img_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read input image: {image_path}")
    if img_bgr.shape[0] != args.height or img_bgr.shape[1] != args.width:
        img_bgr = cv2.resize(img_bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)

    amps = {}
    for channel in channels:
        image = img_bgr[:, :, channel].astype(np.float32) / 255.0
        amp = np.sqrt(image)
        amps[channel] = torch.tensor(amp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return amps


def make_depth_tensor(args, layer_num, device):
    if layer_num == 1:
        return torch.zeros((1, 1, args.height, args.width), dtype=torch.float32, device=device)

    if args.depth_image is not None:
        depth_path = resolve_path(args.depth_image)
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Cannot read depth image: {depth_path}")
        if depth_raw.ndim > 2 and depth_raw.shape[2] > 1:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        if depth_raw.shape[0] != args.height or depth_raw.shape[1] != args.width:
            depth_raw = cv2.resize(depth_raw, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        depth = depth_raw.astype(np.float32) / 255.0
        if Hyperparams.DEPTH_INVERSION:
            depth = 1.0 - depth
        for dep in range(layer_num):
            depth[(depth >= dep / layer_num) & (depth <= (dep + 1) / layer_num)] = dep / layer_num
        return torch.tensor(depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # Synthetic layered depth: horizontal bands covering all layers. This avoids
    # disk I/O and keeps the L2RM mask/update path identical to evaluation.
    rows = torch.arange(args.height, device=device, dtype=torch.int64)
    layer_ids = torch.div(rows * layer_num, args.height, rounding_mode="floor").clamp(max=layer_num - 1)
    depth = (layer_ids.float() / float(layer_num)).view(1, 1, args.height, 1)
    return depth.expand(1, 1, args.height, args.width).contiguous()


def structural_global(device):
    return torch.tensor(1.0, dtype=torch.complex64, device=device)


def structured_l2rm(optics, image_amp_slm, image_depth_slm, device):
    phase_init = torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs
    u = torch.polar(image_amp_slm, phase_init)
    u0 = torch.polar(image_amp_slm, phase_init)
    h_backward_g = structural_global(device)
    h_backward_delta_g = structural_global(device)

    depth_num = optics.layer_num
    for depth_index in range(depth_num):
        if depth_index != 0:
            depth_value = (depth_num - depth_index - 1) / depth_num
            u[image_depth_slm == depth_value] = u0[image_depth_slm == depth_value]
        if depth_index == depth_num - 1:
            u = optics.prop_asm(optics.h_backward_s, h_backward_g, u0=u)
        else:
            u = optics.prop_asm(optics.h_backward_delta_s, h_backward_delta_g, u0=u)

    return u / torch.max(torch.abs(u))


def phase_postprocess(phase, phase_grating):
    phase = phase - phase_grating
    return torch.remainder(phase - phase.mean() + np.pi, 2 * np.pi) / (2 * np.pi)


def elapsed_ms(start, end):
    return start.elapsed_time(end)


def synthesize_cuda_event(model, optics_by_channel, amps_by_channel, depth_slm, channels, device, measure):
    if not measure:
        for channel in channels:
            u = structured_l2rm(optics_by_channel[channel], amps_by_channel[channel], depth_slm, device)
            phase = model(u)
            _ = phase_postprocess(phase, optics_by_channel[channel].phase_grating)
        return None

    events = []
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    total_start.record()

    for channel in channels:
        l2rm_start = torch.cuda.Event(enable_timing=True)
        l2rm_end = torch.cuda.Event(enable_timing=True)
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        post_start = torch.cuda.Event(enable_timing=True)
        post_end = torch.cuda.Event(enable_timing=True)

        l2rm_start.record()
        u = structured_l2rm(optics_by_channel[channel], amps_by_channel[channel], depth_slm, device)
        l2rm_end.record()

        forward_start.record()
        phase = model(u)
        forward_end.record()

        post_start.record()
        _ = phase_postprocess(phase, optics_by_channel[channel].phase_grating)
        post_end.record()

        events.append((l2rm_start, l2rm_end, forward_start, forward_end, post_start, post_end))

    total_end.record()
    torch.cuda.synchronize(device)

    l2rm_ms = sum(elapsed_ms(item[0], item[1]) for item in events)
    forward_ms = sum(elapsed_ms(item[2], item[3]) for item in events)
    postprocess_ms = sum(elapsed_ms(item[4], item[5]) for item in events)
    total_ms = elapsed_ms(total_start, total_end)
    return l2rm_ms, forward_ms, postprocess_ms, total_ms


def timed_cuda_block(device, fn):
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = fn()
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    return result, (end - start) * 1000.0


def synthesize_cuda_sync(model, optics_by_channel, amps_by_channel, depth_slm, channels, device, measure):
    if not measure:
        for channel in channels:
            u = structured_l2rm(optics_by_channel[channel], amps_by_channel[channel], depth_slm, device)
            phase = model(u)
            _ = phase_postprocess(phase, optics_by_channel[channel].phase_grating)
        return None

    torch.cuda.synchronize(device)
    total_start = time.perf_counter()
    l2rm_ms = 0.0
    forward_ms = 0.0
    postprocess_ms = 0.0

    for channel in channels:
        u, elapsed = timed_cuda_block(
            device,
            lambda channel=channel: structured_l2rm(
                optics_by_channel[channel], amps_by_channel[channel], depth_slm, device
            ),
        )
        l2rm_ms += elapsed

        phase, elapsed = timed_cuda_block(device, lambda u=u: model(u))
        forward_ms += elapsed

        _, elapsed = timed_cuda_block(
            device,
            lambda phase=phase, channel=channel: phase_postprocess(phase, optics_by_channel[channel].phase_grating),
        )
        postprocess_ms += elapsed

    torch.cuda.synchronize(device)
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return l2rm_ms, forward_ms, postprocess_ms, total_ms


def time_block_cpu(fn):
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return result, (end - start) * 1000.0


def synthesize_cpu(model, optics_by_channel, amps_by_channel, depth_slm, channels, device, measure):
    if not measure:
        for channel in channels:
            u = structured_l2rm(optics_by_channel[channel], amps_by_channel[channel], depth_slm, device)
            phase = model(u)
            _ = phase_postprocess(phase, optics_by_channel[channel].phase_grating)
        return None

    l2rm_ms = 0.0
    forward_ms = 0.0
    postprocess_ms = 0.0
    total_start = time.perf_counter()
    for channel in channels:
        u, elapsed = time_block_cpu(
            lambda: structured_l2rm(optics_by_channel[channel], amps_by_channel[channel], depth_slm, device)
        )
        l2rm_ms += elapsed
        phase, elapsed = time_block_cpu(lambda: model(u))
        forward_ms += elapsed
        _, elapsed = time_block_cpu(lambda: phase_postprocess(phase, optics_by_channel[channel].phase_grating))
        postprocess_ms += elapsed
    total_ms = (time.perf_counter() - total_start) * 1000.0
    return l2rm_ms, forward_ms, postprocess_ms, total_ms


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "std_ms": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "median_ms": float(np.median(array)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def run_task(task_name, model, device, args):
    task = TASKS[task_name]
    layer_num = task["layers"]
    channels = task["channels"]
    channel_text = "".join(CHANNEL_NAMES[channel] for channel in channels)

    print(f"\n=== {task_name}: layers={layer_num}, channels={channel_text} ===")
    print("Initializing optics and target tensors; this is not included in runtime.")
    optics_by_channel = {channel: make_optics(channel, layer_num, args) for channel in channels}
    amps_by_channel = load_image_channels(args, device, channels)
    depth_slm = make_depth_tensor(args, layer_num, device)

    if device.type == "cuda":
        synthesize = synthesize_cuda_sync if args.timing_mode == "sync" else synthesize_cuda_event
    else:
        synthesize = synthesize_cpu

    with torch.inference_mode():
        for _ in range(args.warmup):
            synthesize(model, optics_by_channel, amps_by_channel, depth_slm, channels, device, measure=False)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        rows = []
        for run_index in range(args.runs):
            l2rm_ms, forward_ms, postprocess_ms, total_ms = synthesize(
                model, optics_by_channel, amps_by_channel, depth_slm, channels, device, measure=True
            )
            rows.append({
                "task": task_name,
                "layers": layer_num,
                "channels": channel_text,
                "run_index": run_index,
                "structured_l2rm_ms": l2rm_ms,
                "encoder_forward_ms": forward_ms,
                "postprocess_ms": postprocess_ms,
                "total_synthesis_ms": total_ms,
            })

    peak_mem_gb = 0.0
    if device.type == "cuda":
        peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    summary = {
        "task": task_name,
        "layers": layer_num,
        "channels": channel_text,
        "width": args.width,
        "height": args.height,
        "z_m": args.z,
        "delta_z_m": args.delta_z,
        "warmup": args.warmup,
        "runs": args.runs,
        "structured_l2rm": summarize([row["structured_l2rm_ms"] for row in rows]),
        "encoder_forward": summarize([row["encoder_forward_ms"] for row in rows]),
        "postprocess": summarize([row["postprocess_ms"] for row in rows]),
        "total_synthesis": summarize([row["total_synthesis_ms"] for row in rows]),
        "fps": 1000.0 / np.mean([row["total_synthesis_ms"] for row in rows]),
        "peak_memory_gb": peak_mem_gb,
    }

    print(
        f"{task_name}: total {summary['total_synthesis']['mean_ms']:.2f} ms, "
        f"L2RM {summary['structured_l2rm']['mean_ms']:.2f} ms, "
        f"forward {summary['encoder_forward']['mean_ms']:.2f} ms, "
        f"post {summary['postprocess']['mean_ms']:.2f} ms, "
        f"FPS {summary['fps']:.2f}, peak {peak_mem_gb:.2f} GB"
    )

    del optics_by_channel, amps_by_channel, depth_slm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, summary


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def flatten_summary(summary):
    row = {
        "task": summary["task"],
        "layers": summary["layers"],
        "channels": summary["channels"],
        "width": summary["width"],
        "height": summary["height"],
        "z_m": summary["z_m"],
        "delta_z_m": summary["delta_z_m"],
        "warmup": summary["warmup"],
        "runs": summary["runs"],
        "fps": summary["fps"],
        "peak_memory_gb": summary["peak_memory_gb"],
    }
    for name in ["structured_l2rm", "encoder_forward", "postprocess", "total_synthesis"]:
        for key, value in summary[name].items():
            row[f"{name}_{key}"] = value
    return row


def run(args):
    device = select_device(args.device)
    model_path = resolve_path(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_path": str(model_path),
        "output_dir": str(output_dir),
        "device": str(device),
        "width": args.width,
        "height": args.height,
        "z_m": args.z,
        "delta_z_m": args.delta_z,
        "pitch_m": args.pitch,
        "factor": args.factor,
        "grating_type": args.grating_type,
        "linear_conv": not args.no_linear_conv,
        "band_limit": not args.no_band_limit,
        "tasks": args.tasks,
        "warmup": args.warmup,
        "runs": args.runs,
        "input_image": str(resolve_path(args.input_image)) if args.input_image else None,
        "depth_image": str(resolve_path(args.depth_image)) if args.depth_image else None,
        "timing_mode": args.timing_mode if device.type == "cuda" else "cpu",
        "notes": [
            "Timing excludes disk I/O, image saving, reconstruction, and metrics.",
            "Propagation kernels are precomputed during Optics initialization and excluded from runtime.",
            "Structured propagation uses global phase set to one, matching the SGWN-Encoder structural evaluation path.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Warmup: {args.warmup} | Runs: {args.runs}")
    print("Kernel initialization is excluded from measured runtime.")

    model = load_model(model_path, device)
    try:
        all_rows = []
        summaries = []
        for task_name in args.tasks:
            rows, summary = run_task(task_name, model, device, args)
            all_rows.extend(rows)
            summaries.append(summary)
            write_csv(output_dir / f"{task_name}_runs.csv", rows)
            (output_dir / f"{task_name}_summary.json").write_text(
                json.dumps(summary, indent=2),
                encoding="utf-8",
            )

        write_csv(output_dir / "timing_runs.csv", all_rows)
        write_csv(output_dir / "timing_summary.csv", [flatten_summary(summary) for summary in summaries])
        (output_dir / "timing_summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SGWN-Encoder 2D/3D 4K hologram synthesis runtime.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--z", type=float, default=0.1)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--tasks", nargs="+", choices=sorted(TASKS), default=["2d_single", "2d_rgb", "3d8_rgb", "3d64_rgb"])
    parser.add_argument("--input-image", default=None, help="Optional RGB image. Loaded once before timing.")
    parser.add_argument("--depth-image", default=None, help="Optional depth map. Loaded once before timing.")
    parser.add_argument("--timing-mode", choices=["sync", "event"], default="sync",
                        help="CUDA timing method. sync uses wall time around synchronized blocks; event is kept for debugging.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
