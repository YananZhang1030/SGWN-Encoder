#!/usr/bin/env python3
"""
Evaluate SGWN-Encoder seed-stability models over the seed-8 distance list.

Distance list matches Random_Diffraction_Field_Training/predict_rgb_batch.py:
manual distances [0.005, 0.1, 0.2] plus 48 random distances generated with
np.random.seed(8), sorted and deduplicated.
"""

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from Optics import Optics
from hyperparams import Hyperparams
from checkpoint_utils import load_phase_model


DEFAULT_MODEL_ROOT = Path("outputs/seed_stability")
DEFAULT_INPUT_DIR = Path("data/evaldataset_rgb")
DEFAULT_OUTPUT_DIR = Path("outputs/seed_stability_new")
CHANNEL_NAMES = "BGR"



def build_distance_list():
    manual_distances = [0.005, 0.1, 0.2]
    np.random.seed(8)
    random_distances = np.random.uniform(low=0.0, high=0.5, size=48).tolist()
    return sorted(list(set(manual_distances + random_distances)))


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parent / path


def discover_seed_models(model_root, seeds, checkpoint):
    model_root = resolve_path(model_root)
    models = []
    for seed in seeds:
        model_path = model_root / f"DF_D_R_90_90_L_350_350_seed{seed}" / "models" / checkpoint
        if not model_path.exists():
            raise FileNotFoundError(f"Missing seed {seed} model: {model_path}")
        models.append((seed, model_path))
    return models


def find_image_pairs(input_dir, allow_missing_depth=False):
    input_dir = resolve_path(input_dir)
    pairs = []
    missing_depth = []
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue
        if path.stem.endswith("_depth"):
            continue
        depth_path = path.with_name(f"{path.stem}_depth.png")
        if depth_path.exists():
            pairs.append((path, depth_path))
        elif allow_missing_depth:
            pairs.append((path, None))
            missing_depth.append(path.name)
    if not pairs:
        raise FileNotFoundError(f"No RGB images found in {input_dir}")
    if missing_depth:
        print(f"Using flat depth for {len(missing_depth)} images without *_depth.png files.")
    return pairs


def select_device(device_name):
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
    Hyperparams.device = device
    Hyperparams.SAVE = True
    Hyperparams.MUL_SAVE = False
    Hyperparams.TIME = False
    return device


def load_model(model_path, device):
    return load_phase_model(model_path, device=device)


def make_optics(first_z, channel, args):
    return Optics(
        channel=channel,
        first_z=first_z,
        delta_z=args.delta_z,
        layer_num=args.layer_num,
        factor=args.factor,
        grating_type=args.grating_type,
        LCoS_res_h=args.height,
        LCoS_res_w=args.width,
        LCoS_pitch=args.pitch,
        linear_conv=not args.no_linear_conv,
        band_limit=not args.no_band_limit,
    )


def load_target_tensors(img_path, depth_path, optics, device):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = img_bgr[:, :, optics.channel]

    if depth_path is None:
        image_depth = np.zeros(img.shape, dtype=np.uint8)
    else:
        depth_raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            raise FileNotFoundError(f"Cannot read depth image: {depth_path}")
        if depth_raw.ndim > 2 and depth_raw.shape[2] > 1:
            image_depth = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        else:
            image_depth = depth_raw

    if img.shape[0] != optics.LCoS_res_h or img.shape[1] != optics.LCoS_res_w:
        img = cv2.resize(img, (optics.LCoS_res_w, optics.LCoS_res_h))
    if image_depth.shape[0] != optics.LCoS_res_h or image_depth.shape[1] != optics.LCoS_res_w:
        image_depth = cv2.resize(image_depth, (optics.LCoS_res_w, optics.LCoS_res_h))

    image = img.astype(np.float32) / 255.0
    image_amp = np.sqrt(image)
    image_amp_slm = torch.tensor(image_amp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    image_ints_ts = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    image_depth = image_depth.astype(np.float32) / 255.0
    if Hyperparams.DEPTH_INVERSION:
        image_depth = 1.0 - image_depth
    depth_num = optics.layer_num
    for dep in range(depth_num):
        image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
    image_depth_slm = torch.tensor(image_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    return img, image_ints_ts, image_amp_slm, image_depth_slm


def phase_to_uint8(phase):
    max_phase = 2 * np.pi
    output_phase = ((phase - phase.mean() + max_phase / 2) % max_phase) / max_phase
    return (output_phase[0, 0] * 255).round().detach().cpu().numpy().clip(0, 255).astype(np.uint8)


def reconstruct_one_channel(model, optics, img_path, depth_path, holo_path, rec_path, use_structural, device):
    img, image_ints_ts, image_amp_slm, image_depth_slm = load_target_tensors(img_path, depth_path, optics, device)
    depth_num = optics.layer_num

    u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)
    u0 = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)

    if use_structural:
        h_backward_g = torch.tensor(1.0, dtype=torch.complex64, device=device)
        h_backward_delta_g = torch.tensor(1.0, dtype=torch.complex64, device=device)
    else:
        h_backward_g = optics.h_backward_g
        h_backward_delta_g = optics.h_backward_delta_g

    for depth_index in range(depth_num):
        if depth_index != 0:
            depth_value = (depth_num - depth_index - 1) / depth_num
            u[image_depth_slm == depth_value] = u0[image_depth_slm == depth_value]
        if depth_index == depth_num - 1:
            u = optics.prop_asm(optics.h_backward_s, h_backward_g, u0=u)
        else:
            u = optics.prop_asm(optics.h_backward_delta_s, h_backward_delta_g, u0=u)

    u = u / torch.max(torch.abs(u))
    phs_slm = model(u)

    os.makedirs(os.path.dirname(holo_path), exist_ok=True)
    cv2.imwrite(str(holo_path), phase_to_uint8(phs_slm))

    phs_slm = phs_slm - optics.phase_grating
    rec_amp_slm = torch.zeros_like(image_amp_slm)
    rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))

    if use_structural:
        h_forward_g = torch.tensor(1.0, dtype=torch.complex64, device=device)
        h_forward_delta_g = torch.tensor(1.0, dtype=torch.complex64, device=device)
    else:
        h_forward_g = optics.h_forward_g
        h_forward_delta_g = optics.h_forward_delta_g

    for dep in range(depth_num):
        if dep == 0:
            rec_u = optics.prop_asm(optics.h_forward_s, h_forward_g, phs_in=phs_slm)
        else:
            rec_u = optics.prop_asm(optics.h_forward_delta_s, h_forward_delta_g, u0=rec_u)
        depth_value = (depth_num - dep - 1) / depth_num
        rec_amp_slm[image_depth_slm == depth_value] = torch.abs(rec_u)[image_depth_slm == depth_value]

    rec_ints = rec_amp_slm ** 2
    coefficient = torch.sum(image_ints_ts) / torch.sum(rec_ints)
    rec = (rec_ints * coefficient * 255).round().squeeze(0).squeeze(0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(rec_path), rec)
    return img, rec


def evaluate_rgb_image(model, optics_by_channel, img_path, depth_path, output_dir, use_structural, device):
    base_name = img_path.stem
    reconstructed_channels = []
    channel_rows = []

    for channel, channel_name in enumerate(CHANNEL_NAMES):
        holo_path = output_dir / f"{base_name}_hologram_{channel_name}.png"
        rec_path = output_dir / f"{base_name}_reconstruction_{channel_name}.png"
        original_ch, rec_ch = reconstruct_one_channel(
            model,
            optics_by_channel[channel],
            img_path,
            depth_path,
            holo_path,
            rec_path,
            use_structural,
            device,
        )
        reconstructed_channels.append(rec_ch)

        channel_psnr = psnr(original_ch, rec_ch, data_range=255)
        channel_ssim = float(ssim(original_ch, rec_ch, data_range=255))
        channel_rows.append({"channel": channel_name, "psnr": channel_psnr, "ssim": channel_ssim})

    merged_reconstruction = cv2.merge(reconstructed_channels)
    merged_path = output_dir / f"{base_name}_reconstruction_merged.png"
    cv2.imwrite(str(merged_path), merged_reconstruction)

    return {
        "image": img_path.name,
        "psnr": float(np.mean([row["psnr"] for row in channel_rows])),
        "ssim": float(np.mean([row["ssim"] for row in channel_rows])),
        "channels": channel_rows,
    }


def parse_summary(summary_path):
    text = Path(summary_path).read_text(encoding="utf-8")
    psnr_match = re.search(r"Average PSNR:\s*([0-9.]+)", text)
    ssim_match = re.search(r"Average SSIM:\s*([0-9.]+)", text)
    success_match = re.search(r"Success:\s*(\d+)/(\d+)", text)
    if not psnr_match or not ssim_match or not success_match:
        return None
    return {
        "avg_psnr": float(psnr_match.group(1)),
        "avg_ssim": float(ssim_match.group(1)),
        "success_count": int(success_match.group(1)),
        "total_count": int(success_match.group(2)),
    }


def write_distance_summary(summary_path, seed, first_z, results, elapsed):
    psnrs = [r["psnr"] for r in results]
    ssims = [r["ssim"] for r in results]
    avg_psnr = float(np.mean(psnrs)) if psnrs else 0.0
    avg_ssim = float(np.mean(ssims)) if ssims else 0.0

    lines = [
        "SGWN-Encoder seed model distance evaluation",
        f"Seed: {seed}",
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


def append_csv(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_aggregate(output_dir, rows):
    if not rows:
        return
    distances = sorted({row["first_z"] for row in rows})
    aggregate_path = output_dir / "distance_seed_aggregate.csv"
    with aggregate_path.open("w", newline="") as f:
        fieldnames = ["first_z", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "num_seeds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for first_z in distances:
            subset = [row for row in rows if row["first_z"] == first_z]
            writer.writerow({
                "first_z": f"{first_z:.10f}",
                "psnr_mean": float(np.mean([r["avg_psnr"] for r in subset])),
                "psnr_std": float(np.std([r["avg_psnr"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "ssim_mean": float(np.mean([r["avg_ssim"] for r in subset])),
                "ssim_std": float(np.std([r["avg_ssim"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "num_seeds": len(subset),
            })

    per_seed_path = output_dir / "seed_aggregate.csv"
    with per_seed_path.open("w", newline="") as f:
        fieldnames = ["seed", "psnr_mean", "psnr_std", "ssim_mean", "ssim_std", "num_distances"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seed in sorted({row["seed"] for row in rows}):
            subset = [row for row in rows if row["seed"] == seed]
            writer.writerow({
                "seed": seed,
                "psnr_mean": float(np.mean([r["avg_psnr"] for r in subset])),
                "psnr_std": float(np.std([r["avg_psnr"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "ssim_mean": float(np.mean([r["avg_ssim"] for r in subset])),
                "ssim_std": float(np.std([r["avg_ssim"] for r in subset], ddof=1)) if len(subset) > 1 else 0.0,
                "num_distances": len(subset),
            })


def run(args):
    device = select_device(args.device)
    input_dir = resolve_path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    distances = build_distance_list()
    if args.max_distances is not None:
        distances = distances[: args.max_distances]
    if args.only_distances:
        selected = {round(float(v), 4) for v in args.only_distances}
        distances = [d for d in distances if round(float(d), 4) in selected]

    models = discover_seed_models(args.model_root, args.seeds, args.checkpoint)
    image_pairs = find_image_pairs(input_dir, allow_missing_depth=args.allow_missing_depth)
    if args.max_images is not None:
        image_pairs = image_pairs[: args.max_images]

    manifest = {
        "device": str(device),
        "distance_seed": 8,
        "manual_distances": [0.005, 0.1, 0.2],
        "num_random_distances": 48,
        "distance_range": [0.0, 0.5],
        "distances": distances,
        "model_paths": {str(seed): str(path) for seed, path in models},
        "input_dir": str(input_dir),
        "allow_missing_depth": args.allow_missing_depth,
        "max_images": args.max_images,
        "use_structural_propagation": args.use_structural_propagation,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_summary_path = output_dir / "all_seed_distance_summary.csv"
    fieldnames = ["seed", "first_z", "z_dir", "avg_psnr", "avg_ssim", "success_count", "total_count", "elapsed_seconds"]
    per_image_path = output_dir / "per_image_seed_distance_metrics.csv"
    per_image_fieldnames = ["seed", "first_z", "image", "psnr", "ssim"]
    all_rows = []

    print(f"Device: {device}")
    print(f"Models: {len(models)} | Distances: {len(distances)} | Images: {len(image_pairs)}")
    print(f"Output: {output_dir}")

    loaded_models = []
    for seed, model_path in models:
        print(f"Loading seed {seed}: {model_path}")
        loaded_models.append((seed, model_path, load_model(model_path, device)))

    try:
        for first_z in distances:
            z_name = f"z_{first_z:.4f}"
            pending = []
            for seed, model_path, model in loaded_models:
                z_dir = output_dir / f"seed{seed}" / z_name
                summary_path = z_dir / "summary.txt"
                if summary_path.exists() and not args.overwrite:
                    parsed = parse_summary(summary_path)
                    if parsed is not None:
                        row = {
                            "seed": seed,
                            "first_z": float(first_z),
                            "z_dir": str(z_dir),
                            "avg_psnr": parsed["avg_psnr"],
                            "avg_ssim": parsed["avg_ssim"],
                            "success_count": parsed["success_count"],
                            "total_count": parsed["total_count"],
                            "elapsed_seconds": 0.0,
                        }
                        all_rows.append(row)
                        print(f"Skip seed {seed} {z_name}: PSNR {parsed['avg_psnr']:.4f}, SSIM {parsed['avg_ssim']:.4f}")
                        continue
                pending.append((seed, model_path, model, z_dir, summary_path))

            if not pending:
                continue

            print(f"\n=== Distance {z_name}: initializing optics once for {len(pending)} seed model(s) ===")
            optics_by_channel = {channel: make_optics(first_z, channel, args) for channel in range(3)}

            for seed, model_path, model, z_dir, summary_path in pending:
                z_dir.mkdir(parents=True, exist_ok=True)
                start_time = time.time()
                print(f"Seed {seed} | {z_name}: evaluating")
                results = []
                with torch.inference_mode():
                    for index, (img_path, depth_path) in enumerate(image_pairs, 1):
                        print(f"  [{index}/{len(image_pairs)}] {img_path.name}")
                        results.append(
                            evaluate_rgb_image(
                                model,
                                optics_by_channel,
                                img_path,
                                depth_path,
                                z_dir,
                                args.use_structural_propagation,
                                device,
                            )
                        )
                elapsed = time.time() - start_time
                avg_psnr, avg_ssim = write_distance_summary(summary_path, seed, first_z, results, elapsed)
                for result in results:
                    append_csv(per_image_path, per_image_fieldnames, {
                        "seed": seed,
                        "first_z": float(first_z),
                        "image": result["image"],
                        "psnr": result["psnr"],
                        "ssim": result["ssim"],
                    })
                row = {
                    "seed": seed,
                    "first_z": float(first_z),
                    "z_dir": str(z_dir),
                    "avg_psnr": avg_psnr,
                    "avg_ssim": avg_ssim,
                    "success_count": len(results),
                    "total_count": len(image_pairs),
                    "elapsed_seconds": elapsed,
                }
                all_rows.append(row)
                append_csv(all_summary_path, fieldnames, row)
                print(f"Done seed {seed} {z_name}: PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, {elapsed:.1f}s")

            del optics_by_channel
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for _, _, model in loaded_models:
            del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if all_summary_path.exists():
        with all_summary_path.open("r", newline="") as f:
            existing_rows = list(csv.DictReader(f))
        normalized_rows = []
        for row in existing_rows:
            normalized_rows.append({
                "seed": int(row["seed"]),
                "first_z": float(row["first_z"]),
                "avg_psnr": float(row["avg_psnr"]),
                "avg_ssim": float(row["avg_ssim"]),
            })
        write_aggregate(output_dir, normalized_rows)
    else:
        write_aggregate(output_dir, all_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate five SGWN-Encoder seed models over the np.random.seed(8) distance list")
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoint", default="model_state_dict.pt")
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--use-structural-propagation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-distances", type=int, default=None)
    parser.add_argument("--only-distances", type=float, nargs="+", default=None)
    parser.add_argument("--allow-missing-depth", action="store_true",
                        help="Evaluate RGB-only datasets with a flat zero depth map. Valid for layer_num=1 comparisons.")
    parser.add_argument("--max-images", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
