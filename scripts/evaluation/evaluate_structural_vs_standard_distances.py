#!/usr/bin/env python3
"""
Compare structural propagation with standard ASM propagation over a shared
distance list.

Default distance setup:
- manual distances: [0.005, 0.1, 0.2] m
- random distances: 17 samples from [0.0, 0.2] m
- random seed: 8

The two propagation modes use the same model, images, depth maps, optics
settings, and distance list. Results are written as per-distance and aggregate
CSV files, including standard-minus-structural PSNR/SSIM deltas.
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

import numpy as np
import torch

import evaluate_dataset_model_distances as evaluator


MANUAL_DISTANCE_LIST = [0.005, 0.1, 0.2]
NUM_RANDOM_DISTANCES = 17
DISTANCE_RANGE = [0.0, 0.2]
DISTANCE_SEED = 8

DEFAULT_MODEL_ROOT = Path("checkpoints")
DEFAULT_INPUT_DIR = Path("data/evaldataset_rgb")
DEFAULT_OUTPUT_DIR = Path("outputs/structural_vs_standard_distance_comparison")
PROPAGATION_MODES = {
    "structural": True,
    "standard": False,
}


def build_distance_list(manual_distances, num_random_distances, distance_range, distance_seed):
    manual = [float(distance) for distance in manual_distances]
    np.random.seed(int(distance_seed))
    random_distances = np.random.uniform(
        low=float(distance_range[0]),
        high=float(distance_range[1]),
        size=int(num_random_distances),
    ).tolist()
    return sorted(set(manual + random_distances))


def write_distance_summary(summary_path, model_name, propagation_mode, first_z, results, elapsed):
    psnrs = [row["psnr"] for row in results]
    ssims = [row["ssim"] for row in results]
    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    avg_ssim = float(np.mean(ssims)) if ssims else 0.0

    lines = [
        "SGWN-Encoder structural-vs-standard propagation distance evaluation",
        f"Model: {model_name}",
        f"Propagation mode: {propagation_mode}",
        f"first_z: {first_z}",
        f"Success: {len(results)}/{len(results)}",
        f"Elapsed seconds: {elapsed:.2f}",
        "",
        f"Average PSNR: {avg_psnr:.4f}",
        f"Average SSIM: {avg_ssim:.4f}",
        "",
        "Per-image results:",
    ]
    for result in results:
        lines.append(f"{result['image']}: PSNR={result['psnr']:.4f}, SSIM={result['ssim']:.4f}")

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return avg_psnr, avg_ssim


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregates(output_dir, rows):
    if not rows:
        return

    aggregate_rows = []
    model_names = sorted({row["model_name"] for row in rows})
    modes = sorted({row["propagation_mode"] for row in rows})

    for model_name in model_names:
        for mode in modes:
            subset = [
                row for row in rows
                if row["model_name"] == model_name and row["propagation_mode"] == mode
            ]
            if not subset:
                continue
            aggregate_rows.append({
                "model_name": model_name,
                "propagation_mode": mode,
                "psnr_mean": float(np.mean([row["avg_psnr"] for row in subset])),
                "psnr_std": float(np.std([row["avg_psnr"] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "ssim_mean": float(np.mean([row["avg_ssim"] for row in subset])),
                "ssim_std": float(np.std([row["avg_ssim"] for row in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "num_distances": len(subset),
            })

    write_csv(
        output_dir / "mode_aggregate.csv",
        ["model_name", "propagation_mode", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "num_distances"],
        aggregate_rows,
    )

    comparison_rows = []
    distances = sorted({row["first_z"] for row in rows})
    for model_name in model_names:
        for first_z in distances:
            by_mode = {
                row["propagation_mode"]: row
                for row in rows
                if row["model_name"] == model_name and row["first_z"] == first_z
            }
            comparison_row = {
                "model_name": model_name,
                "first_z": f"{first_z:.10f}",
            }
            for mode in ("structural", "standard"):
                row = by_mode.get(mode)
                comparison_row[f"{mode}_psnr"] = row["avg_psnr"] if row else ""
                comparison_row[f"{mode}_ssim"] = row["avg_ssim"] if row else ""
            if "structural" in by_mode and "standard" in by_mode:
                comparison_row["psnr_diff_standard_minus_structural"] = (
                    by_mode["standard"]["avg_psnr"] - by_mode["structural"]["avg_psnr"]
                )
                comparison_row["ssim_diff_standard_minus_structural"] = (
                    by_mode["standard"]["avg_ssim"] - by_mode["structural"]["avg_ssim"]
                )
            else:
                comparison_row["psnr_diff_standard_minus_structural"] = ""
                comparison_row["ssim_diff_standard_minus_structural"] = ""
            comparison_rows.append(comparison_row)

    write_csv(
        output_dir / "mode_comparison_by_distance.csv",
        [
            "model_name",
            "first_z",
            "structural_psnr",
            "structural_ssim",
            "standard_psnr",
            "standard_ssim",
            "psnr_diff_standard_minus_structural",
            "ssim_diff_standard_minus_structural",
        ],
        comparison_rows,
    )


def run(args):
    device = evaluator.select_device(args.device)
    input_dir = evaluator.resolve_path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances = build_distance_list(
        args.manual_distances,
        args.num_random_distances,
        args.distance_range,
        args.distance_seed,
    )
    if args.max_distances is not None:
        distances = distances[: args.max_distances]
    if args.only_distances:
        selected = {round(float(distance), 4) for distance in args.only_distances}
        distances = [distance for distance in distances if round(float(distance), 4) in selected]

    for mode in args.propagation_modes:
        if mode not in PROPAGATION_MODES:
            raise ValueError(f"Unknown propagation mode: {mode}")

    models = evaluator.discover_dataset_models(args.model_root, args.model_names)
    image_pairs = evaluator.find_image_pairs(input_dir, allow_missing_depth=args.allow_missing_depth)
    if args.max_images is not None:
        image_pairs = image_pairs[: args.max_images]

    manifest = {
        "device": str(device),
        "manual_distances": args.manual_distances,
        "num_random_distances": args.num_random_distances,
        "distance_range": args.distance_range,
        "distance_seed": args.distance_seed,
        "distances": distances,
        "propagation_modes": args.propagation_modes,
        "model_paths": {str(model_name): str(path) for model_name, path in models},
        "input_dir": str(input_dir),
        "allow_missing_depth": args.allow_missing_depth,
        "max_images": args.max_images,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    summary_path = output_dir / "all_mode_distance_summary.csv"
    summary_fields = [
        "propagation_mode",
        "model_name",
        "first_z",
        "z_dir",
        "avg_psnr",
        "avg_ssim",
        "success_count",
        "total_count",
        "elapsed_seconds",
    ]
    per_image_path = output_dir / "per_image_mode_distance_metrics.csv"
    per_image_fields = ["propagation_mode", "model_name", "first_z", "image", "psnr", "ssim"]
    all_rows = []

    print(f"Device: {device}")
    print(f"Modes: {', '.join(args.propagation_modes)}")
    print(f"Models: {len(models)} | Distances: {len(distances)} | Images: {len(image_pairs)}")
    print(f"Output: {output_dir}")

    loaded_models = []
    for model_name, model_path in models:
        print(f"Loading model {model_name}: {model_path}")
        loaded_models.append((model_name, model_path, evaluator.load_model(model_path, device)))

    try:
        for first_z in distances:
            z_name = f"z_{first_z:.4f}"
            print(f"\n=== Distance {z_name}: initializing optics once ===")
            optics_by_channel = {
                channel: evaluator.make_optics(first_z, channel, args)
                for channel in range(3)
            }

            for propagation_mode in args.propagation_modes:
                use_structural = PROPAGATION_MODES[propagation_mode]
                for model_name, _, model in loaded_models:
                    z_dir = output_dir / propagation_mode / model_name / z_name
                    mode_summary_path = z_dir / "summary.txt"
                    if mode_summary_path.exists() and not args.overwrite:
                        parsed = evaluator.parse_summary(mode_summary_path)
                        if parsed is not None:
                            row = {
                                "propagation_mode": propagation_mode,
                                "model_name": model_name,
                                "first_z": float(first_z),
                                "z_dir": str(z_dir),
                                "avg_psnr": parsed["avg_psnr"],
                                "avg_ssim": parsed["avg_ssim"],
                                "success_count": parsed["success_count"],
                                "total_count": parsed["total_count"],
                                "elapsed_seconds": 0.0,
                            }
                            all_rows.append(row)
                            print(
                                f"Skip {propagation_mode} {model_name} {z_name}: "
                                f"PSNR {parsed['avg_psnr']:.4f}, SSIM {parsed['avg_ssim']:.4f}"
                            )
                            continue

                    z_dir.mkdir(parents=True, exist_ok=True)
                    start_time = time.time()
                    print(f"Mode {propagation_mode} | Model {model_name} | {z_name}: evaluating")
                    results = []
                    with torch.inference_mode():
                        for index, (img_path, depth_path) in enumerate(image_pairs, 1):
                            print(f"  [{index}/{len(image_pairs)}] {img_path.name}")
                            results.append(
                                evaluator.evaluate_rgb_image(
                                    model,
                                    optics_by_channel,
                                    img_path,
                                    depth_path,
                                    z_dir,
                                    use_structural,
                                    device,
                                )
                            )
                    elapsed = time.time() - start_time
                    avg_psnr, avg_ssim = write_distance_summary(
                        mode_summary_path,
                        model_name,
                        propagation_mode,
                        first_z,
                        results,
                        elapsed,
                    )
                    for result in results:
                        evaluator.append_csv(per_image_path, per_image_fields, {
                            "propagation_mode": propagation_mode,
                            "model_name": model_name,
                            "first_z": float(first_z),
                            "image": result["image"],
                            "psnr": result["psnr"],
                            "ssim": result["ssim"],
                        })

                    row = {
                        "propagation_mode": propagation_mode,
                        "model_name": model_name,
                        "first_z": float(first_z),
                        "z_dir": str(z_dir),
                        "avg_psnr": avg_psnr,
                        "avg_ssim": avg_ssim,
                        "success_count": len(results),
                        "total_count": len(image_pairs),
                        "elapsed_seconds": elapsed,
                    }
                    all_rows.append(row)
                    evaluator.append_csv(summary_path, summary_fields, row)
                    print(
                        f"Done {propagation_mode} {model_name} {z_name}: "
                        f"PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, {elapsed:.1f}s"
                    )

            del optics_by_channel
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for _, _, model in loaded_models:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if summary_path.exists():
        with summary_path.open("r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        all_rows = [
            {
                "propagation_mode": row["propagation_mode"],
                "model_name": row["model_name"],
                "first_z": float(row["first_z"]),
                "avg_psnr": float(row["avg_psnr"]),
                "avg_ssim": float(row["avg_ssim"]),
            }
            for row in existing_rows
        ]
    write_aggregates(output_dir, all_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare structural and standard ASM propagation over a configurable distance list"
    )
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-names", nargs="+", default=["sgwn_encoder_dataset_free"])
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--manual-distances", type=float, nargs="+", default=MANUAL_DISTANCE_LIST)
    parser.add_argument("--num-random-distances", type=int, default=NUM_RANDOM_DISTANCES)
    parser.add_argument("--distance-range", type=float, nargs=2, default=DISTANCE_RANGE)
    parser.add_argument("--distance-seed", type=int, default=DISTANCE_SEED)
    parser.add_argument("--propagation-modes", nargs="+", choices=sorted(PROPAGATION_MODES), default=["structural", "standard"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-distances", type=int, default=None)
    parser.add_argument("--only-distances", type=float, nargs="+", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument(
        "--allow-missing-depth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Evaluate RGB-only datasets with a flat zero depth map. "
            "Enabled by default because the default evaldataset_rgb has no depth maps; "
            "use --no-allow-missing-depth to require *_depth.png files."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
