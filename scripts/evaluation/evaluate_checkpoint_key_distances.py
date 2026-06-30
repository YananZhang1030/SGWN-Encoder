#!/usr/bin/env python3
"""
Evaluate one SGWN-Encoder checkpoint on DIV2K valid HR at 4K.

Default experiment:
- model: checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt
- dataset: data/DIV2K_valid_HR
- size: 4k:3840x2160
- distances: 0.005, 0.1, 0.2 m

The script records per-image and per-distance PSNR/SSIM. By default it does not
save hologram/reconstruction PNGs, which keeps the large-test-set run practical.
"""

import argparse
import csv
import json
import os
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


DEFAULT_MODEL_PATH = Path("checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt")
DEFAULT_INPUT_DIR = Path("data/DIV2K_valid_HR")
DEFAULT_OUTPUT_DIR = Path("outputs/checkpoint_DIV2K_valid_HR_keydist_4k")
CHANNEL_NAMES = "BGR"


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def find_images(input_dir, max_images=None):
    input_dir = resolve_path(input_dir)
    images = [
        path for path in sorted(input_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        and not path.stem.endswith("_depth")
    ]
    if not images:
        raise FileNotFoundError(f"No RGB images found in {input_dir}")
    if max_images is not None:
        images = images[:max_images]
    return images


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


def make_optics(first_z, channel, width, height, args):
    return Optics(
        channel=channel,
        first_z=first_z,
        delta_z=args.delta_z,
        layer_num=args.layer_num,
        factor=args.factor,
        grating_type=args.grating_type,
        LCoS_res_h=height,
        LCoS_res_w=width,
        LCoS_pitch=args.pitch,
        linear_conv=not args.no_linear_conv,
        band_limit=not args.no_band_limit,
    )


def load_channel_tensors(img_path, channel, width, height, device):
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img = img_bgr[:, :, channel]
    if img.shape[0] != height or img.shape[1] != width:
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    image = img.astype(np.float32) / 255.0
    image_amp = np.sqrt(image)
    image_amp_slm = torch.tensor(image_amp, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    image_ints_ts = torch.tensor(image, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    # DIV2K_valid_HR is RGB-only; for layer_num=1 this flat depth selects the only plane.
    image_depth = np.zeros((height, width), dtype=np.float32)
    if Hyperparams.DEPTH_INVERSION:
        image_depth = 1.0 - image_depth
    depth_num = 1
    image_depth[(image_depth >= 0.0) & (image_depth <= 1.0)] = 0.0
    image_depth_slm = torch.tensor(image_depth, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

    return img, image_ints_ts, image_amp_slm, image_depth_slm


def phase_to_uint8(phase):
    max_phase = 2 * np.pi
    output_phase = ((phase - phase.mean() + max_phase / 2) % max_phase) / max_phase
    return (output_phase[0, 0] * 255).round().detach().cpu().numpy().clip(0, 255).astype(np.uint8)


def reconstruct_one_channel(model, optics, img_path, channel, width, height, save_dir, save_images, use_structural, device):
    original, image_ints_ts, image_amp_slm, image_depth_slm = load_channel_tensors(
        img_path, channel, width, height, device
    )
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

    if save_images:
        save_dir.mkdir(parents=True, exist_ok=True)
        channel_name = CHANNEL_NAMES[channel]
        cv2.imwrite(str(save_dir / f"{img_path.stem}_hologram_{channel_name}.png"), phase_to_uint8(phs_slm))

    phs_slm = phs_slm - optics.phase_grating
    rec_amp_slm = torch.zeros_like(image_amp_slm)

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

    if save_images:
        channel_name = CHANNEL_NAMES[channel]
        cv2.imwrite(str(save_dir / f"{img_path.stem}_reconstruction_{channel_name}.png"), rec)

    return original, rec


def evaluate_image(model, optics_by_channel, img_path, width, height, save_dir, save_images, use_structural, device):
    reconstructed_channels = []
    channel_rows = []
    for channel, channel_name in enumerate(CHANNEL_NAMES):
        original_ch, rec_ch = reconstruct_one_channel(
            model,
            optics_by_channel[channel],
            img_path,
            channel,
            width,
            height,
            save_dir,
            save_images,
            use_structural,
            device,
        )
        reconstructed_channels.append(rec_ch)
        channel_rows.append({
            "channel": channel_name,
            "psnr": float(psnr(original_ch, rec_ch, data_range=255)),
            "ssim": float(ssim(original_ch, rec_ch, data_range=255)),
        })

    if save_images:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"{img_path.stem}_reconstruction_merged.png"), cv2.merge(reconstructed_channels))

    return {
        "image": img_path.name,
        "psnr": float(np.mean([row["psnr"] for row in channel_rows])),
        "ssim": float(np.mean([row["ssim"] for row in channel_rows])),
        "channels": channel_rows,
    }


def append_csv(path, fieldnames, row):
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_csv_rows(path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_size(value):
    label, dims = value.split(":", 1) if ":" in value else (value, value)
    width_text, height_text = dims.lower().split("x", 1)
    return label, int(width_text), int(height_text)


def metric_key(size_label, first_z, image_name):
    return (size_label, f"{float(first_z):.12g}", image_name)


def row_key(row):
    return metric_key(row["size_label"], row["first_z"], row["image"])


def expected_image_outputs(save_dir, image_stem):
    outputs = [save_dir / f"{image_stem}_reconstruction_merged.png"]
    for channel_name in CHANNEL_NAMES:
        outputs.append(save_dir / f"{image_stem}_hologram_{channel_name}.png")
        outputs.append(save_dir / f"{image_stem}_reconstruction_{channel_name}.png")
    return outputs


def outputs_exist(save_dir, image_stem, save_images):
    if not save_images:
        return True
    return all(path.exists() for path in expected_image_outputs(save_dir, image_stem))


def load_completed_metrics(per_image_path, save_images, output_dir):
    completed_rows = {}
    for row in read_csv_rows(per_image_path):
        save_dir = output_dir / row["size_label"] / f"z_{float(row['first_z']):.4f}"
        image_stem = Path(row["image"]).stem
        if outputs_exist(save_dir, image_stem, save_images):
            completed_rows[row_key(row)] = row
    return completed_rows


def make_per_image_row(label, width, height, first_z, result, elapsed_seconds):
    return {
        "size_label": label,
        "width": width,
        "height": height,
        "first_z": float(first_z),
        "image": result["image"],
        "psnr": result["psnr"],
        "ssim": result["ssim"],
        "elapsed_seconds": elapsed_seconds,
    }


def summarize_distance(label, width, height, first_z, rows, elapsed_seconds):
    return {
        "size_label": label,
        "width": width,
        "height": height,
        "first_z": float(first_z),
        "avg_psnr": float(np.mean([float(row["psnr"]) for row in rows])) if rows else 0.0,
        "avg_ssim": float(np.mean([float(row["ssim"]) for row in rows])) if rows else 0.0,
        "num_images": len(rows),
        "elapsed_seconds": elapsed_seconds,
    }


def write_summaries(output_dir, per_image_rows, distance_fields):
    grouped = {}
    for row in per_image_rows:
        key = (row["size_label"], int(row["width"]), int(row["height"]), f"{float(row['first_z']):.12g}")
        grouped.setdefault(key, []).append(row)

    distance_rows = []
    for key in sorted(grouped, key=lambda item: (item[0], float(item[3]))):
        label, width, height, first_z_text = key
        subset = grouped[key]
        elapsed = float(np.sum([float(row.get("elapsed_seconds") or 0.0) for row in subset]))
        distance_rows.append(summarize_distance(label, width, height, float(first_z_text), subset, elapsed))

    write_csv(output_dir / "distance_summary.csv", distance_fields, distance_rows)

    size_rows = []
    size_keys = sorted({(row["size_label"], int(row["width"]), int(row["height"])) for row in distance_rows})
    for label, width, height in size_keys:
        subset = [row for row in distance_rows if row["size_label"] == label]
        if not subset:
            continue
        size_rows.append({
            "size_label": label,
            "width": width,
            "height": height,
            "psnr_mean_over_distances": float(np.mean([row["avg_psnr"] for row in subset])),
            "ssim_mean_over_distances": float(np.mean([row["avg_ssim"] for row in subset])),
            "num_distances": len(subset),
        })
    write_csv(
        output_dir / "size_summary.csv",
        ["size_label", "width", "height", "psnr_mean_over_distances", "ssim_mean_over_distances", "num_distances"],
        size_rows,
    )
    return distance_rows


def run(args):
    device = select_device(args.device)
    model_path = resolve_path(args.model_path)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = find_images(args.input_dir, max_images=args.max_images)
    sizes = [parse_size(value) for value in args.sizes]

    manifest = {
        "model_path": str(model_path),
        "input_dir": str(resolve_path(args.input_dir)),
        "output_dir": str(output_dir),
        "device": str(device),
        "sizes": [{"label": label, "width": width, "height": height} for label, width, height in sizes],
        "distances": args.distances,
        "num_images": len(images),
        "save_images": args.save_images,
        "use_structural_propagation": args.use_structural_propagation,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    per_image_path = output_dir / "per_image_metrics.csv"
    distance_summary_path = output_dir / "distance_summary.csv"
    per_image_fields = ["size_label", "width", "height", "first_z", "image", "psnr", "ssim", "elapsed_seconds"]
    distance_fields = ["size_label", "width", "height", "first_z", "avg_psnr", "avg_ssim", "num_images", "elapsed_seconds"]
    completed_rows = load_completed_metrics(per_image_path, args.save_images, output_dir)
    per_image_rows = list(completed_rows.values())

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Images: {len(images)} | Distances: {len(args.distances)} | Sizes: {len(sizes)}")
    print(f"Output: {output_dir}")
    print(f"Save images: {args.save_images}")
    print(f"Completed rows found: {len(completed_rows)}")

    model = load_model(model_path, device)
    try:
        for label, width, height in sizes:
            print(f"\n=== Size {label}: {width}x{height} ===")
            for first_z in args.distances:
                z_name = f"z_{first_z:.4f}"
                print(f"{label} | {z_name}: initializing optics")
                optics_by_channel = {channel: make_optics(first_z, channel, width, height, args) for channel in range(3)}
                results = []
                start_distance = time.time()
                with torch.inference_mode():
                    for index, img_path in enumerate(images, 1):
                        save_dir = output_dir / label / z_name
                        current_key = metric_key(label, first_z, img_path.name)
                        if current_key in completed_rows and outputs_exist(save_dir, img_path.stem, args.save_images):
                            row = completed_rows[current_key]
                            results.append({
                                "image": row["image"],
                                "psnr": float(row["psnr"]),
                                "ssim": float(row["ssim"]),
                            })
                            print(f"  [{index}/{len(images)}] {img_path.name} - skip existing")
                            continue

                        print(f"  [{index}/{len(images)}] {img_path.name}")
                        start_image = time.time()
                        result = evaluate_image(
                            model,
                            optics_by_channel,
                            img_path,
                            width,
                            height,
                            save_dir,
                            args.save_images,
                            args.use_structural_propagation,
                            device,
                        )
                        image_elapsed = time.time() - start_image
                        results.append(result)
                        row = make_per_image_row(label, width, height, first_z, result, image_elapsed)
                        completed_rows[row_key(row)] = row
                        per_image_rows.append(row)
                        append_csv(per_image_path, per_image_fields, row)
                elapsed = time.time() - start_distance
                summary_row = summarize_distance(label, width, height, first_z, results, elapsed)
                avg_psnr = summary_row["avg_psnr"]
                avg_ssim = summary_row["avg_ssim"]
                summary_txt = output_dir / label / z_name / "summary.txt"
                summary_txt.parent.mkdir(parents=True, exist_ok=True)
                summary_txt.write_text(
                    "SGWN-Encoder checkpoint DIV2K valid key-distance evaluation\n"
                    f"Size: {label} ({width}x{height})\n"
                    f"first_z: {first_z}\n"
                    f"Success: {len(results)}/{len(images)}\n"
                    f"Elapsed seconds: {elapsed:.2f}\n\n"
                    f"Average PSNR: {avg_psnr:.4f}\n"
                    f"Average SSIM: {avg_ssim:.4f}\n",
                    encoding="utf-8",
                )
                print(f"Done {label} {z_name}: PSNR {avg_psnr:.4f}, SSIM {avg_ssim:.4f}, {elapsed:.1f}s")
                del optics_by_channel
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_summaries(output_dir, per_image_rows, distance_fields)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one SGWN-Encoder checkpoint on DIV2K valid at 4K key distances")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--sizes", nargs="+", default=["4k:3840x2160"])
    parser.add_argument("--distances", type=float, nargs="+", default=[0.005, 0.1, 0.2])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--use-structural-propagation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action="store_true", help="Also save hologram/reconstruction PNGs for every image.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
