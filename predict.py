#!/usr/bin/env python3
"""Generate phase holograms and numerical reconstructions with SGWN-Encoder."""

import contextlib
import io
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from CNNs import TPN_R
from Optics import Optics
from Trainer import Trainer
from hyperparams import Hyperparams

REPO_ROOT = Path(__file__).resolve().parent
CHANNEL_NAMES = ["B", "G", "R"]

# Prediction configuration
MODEL_PATH = "checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt"
GPU_ID = 0
BATCH_MODE = False
MUL_SAVE = True
SAVE_ASM_INPUT = True
COLORIZE_ASM_INPUT_PHASE = False
MERGE_RGB_RECONSTRUCTION = True
VERBOSE = False

# Propagation mode
USE_STRUCTURAL_PROPAGATION = True

# Optical system configuration
OPTICS_CONFIG = {
    "channel": 1,
    "first_z": 0.005,
    "delta_z": 0.005,
    "layer_num": 4,
    "LCoS_res_h": 2160,
    "LCoS_res_w": 3840,
}

# Path configuration
if BATCH_MODE:
    INPUT_DIR = "eval"
    OUTPUT_DIR = "pred"
else:
    INPUT_IMAGE = "data/example_input/image.png"
    INPUT_DEPTH = "data/example_input/image_depth.png"
    OUTPUT_DIR = "pred"

OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "StructurePropagation" if USE_STRUCTURAL_PROPAGATION else "StandardASM",
)


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def configure_device(gpu_id):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    Hyperparams.device = device
    Hyperparams.SAVE = True
    Hyperparams.MUL_SAVE = MUL_SAVE
    return device


def parse_metrics(output):
    psnr_match = re.search(r"PSNR:\s*([\d.]+)", output)
    ssim_match = re.search(r"SSIM:\s*([\d.]+)", output)
    coeff_match = re.search(r"Coefficient: *(\s*inf|[\d.]+)", output)

    psnr = float(psnr_match.group(1)) if psnr_match else None
    ssim = float(ssim_match.group(1)) if ssim_match else None
    coefficient = None
    if coeff_match:
        coeff_text = coeff_match.group(1).strip()
        coefficient = float("inf") if coeff_text == "inf" else float(coeff_text)
    return psnr, ssim, coefficient


def format_metric(value, suffix=""):
    if value is None:
        return "n/a"
    if value == float("inf"):
        return "inf"
    return f"{value:.4f}{suffix}"


def print_configuration(model_path, output_dir, device):
    mode = "batch" if BATCH_MODE else "single image"
    propagation = "structural" if USE_STRUCTURAL_PROPAGATION else "standard ASM"
    channel = OPTICS_CONFIG["channel"]
    print("Configuration")
    print(f"  model: {model_path}")
    print(f"  mode: {mode}, propagation: {propagation}, device: {device}")
    print(
        f"  optics: channel {channel} ({CHANNEL_NAMES[channel]}), "
        f"{OPTICS_CONFIG['LCoS_res_h']}x{OPTICS_CONFIG['LCoS_res_w']}, "
        f"layers={OPTICS_CONFIG['layer_num']}"
    )
    print(f"  output: {output_dir}")


def save_prediction_config(
    config_path,
    model_path,
    img_path,
    eval_optics,
    phs_path,
    rec_path,
    psnr,
    ssim,
    coefficient,
):
    propagation = "structural" if USE_STRUCTURAL_PROPAGATION else "standard ASM"
    config_info = f"""Prediction Configuration
========================
Model path: {model_path}
Input image: {img_path}
Propagation mode: {propagation}

Optics
------
Channel: {OPTICS_CONFIG['channel']} ({CHANNEL_NAMES[OPTICS_CONFIG['channel']]})
Wavelength: {eval_optics.wavelength:.6e} m
First z: {OPTICS_CONFIG['first_z']}
Delta z: {OPTICS_CONFIG['delta_z']}
Depth layers: {OPTICS_CONFIG['layer_num']}

Outputs
-------
Hologram: {phs_path}
Reconstruction: {rec_path}

Metrics
-------
PSNR: {format_metric(psnr, ' dB')}
SSIM: {format_metric(ssim)}
Coefficient: {format_metric(coefficient)}
"""
    Path(config_path).write_text(config_info, encoding="utf-8")


def predict_single_image(
    model_path,
    img_path,
    depth_path,
    output_dir,
    use_structural_propagation=False,
    save_asm_input=False,
    colorize_asm_input_phase=False,
    output_suffix="",
):
    os.makedirs(output_dir, exist_ok=True)

    base_name = Path(img_path).stem
    output_name = f"{base_name}{output_suffix}"
    phs_path = os.path.join(output_dir, f"{output_name}_hologram.png")
    rec_path = os.path.join(output_dir, f"{output_name}_reconstruction.png")
    mul_rec_path = os.path.join(output_dir, f"{output_name}_multilayer_reconstruction.png")
    config_path = os.path.join(output_dir, f"{output_name}_config.txt")

    eval_optics = Optics(**OPTICS_CONFIG)
    trainer = Trainer(
        eval_optics,
        eval_optics,
        TPN_R(),
        os.path.dirname(model_path),
        PSD_name="/PSD_predict.npy",
    )
    trainer.model_best_path = model_path

    print(f"Predicting {output_name}...")
    try:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            trainer.predict(
                img_path,
                depth_path,
                phs_path,
                rec_path,
                mul_rec_path,
                use_structural_propagation,
                save_asm_input_field=save_asm_input,
                colorize_asm_input_phase=colorize_asm_input_phase,
                output_dir_for_save=output_dir,
                save_prefix=output_name,
            )

        internal_output = captured.getvalue()
        if VERBOSE and internal_output:
            print(internal_output.rstrip())

        psnr, ssim, coefficient = parse_metrics(internal_output)
        save_prediction_config(
            config_path,
            model_path,
            img_path,
            eval_optics,
            phs_path,
            rec_path,
            psnr,
            ssim,
            coefficient,
        )

        print(
            f"Done {output_name}: PSNR={format_metric(psnr, ' dB')}, "
            f"SSIM={format_metric(ssim)}, coeff={format_metric(coefficient)}"
        )
        if VERBOSE:
            print(f"  hologram: {phs_path}")
            print(f"  reconstruction: {rec_path}")
        return True, psnr, ssim
    except Exception as exc:
        print(f"Prediction failed for {output_name}: {exc}")
        if VERBOSE:
            import traceback

            traceback.print_exc()
        return False, None, None


def predict_rgb_merged_image(
    model_path,
    img_path,
    depth_path,
    output_dir,
    use_structural_propagation=False,
    save_asm_input=False,
    colorize_asm_input_phase=False,
):
    original_channel = OPTICS_CONFIG["channel"]
    base_name = Path(img_path).stem
    reconstructed_channels = []
    channel_results = []

    try:
        for channel, channel_name in enumerate(CHANNEL_NAMES):
            OPTICS_CONFIG["channel"] = channel
            success, psnr, ssim = predict_single_image(
                model_path,
                img_path,
                depth_path,
                output_dir,
                use_structural_propagation,
                save_asm_input,
                colorize_asm_input_phase,
                output_suffix=f"_{channel_name}",
            )
            channel_results.append(
                {"channel": channel_name, "success": success, "psnr": psnr, "ssim": ssim}
            )
            if not success:
                return False, None, None

            rec_path = os.path.join(output_dir, f"{base_name}_{channel_name}_reconstruction.png")
            rec_channel = cv2.imread(rec_path, cv2.IMREAD_GRAYSCALE)
            if rec_channel is None:
                raise FileNotFoundError(f"Reconstruction channel not found: {rec_path}")
            reconstructed_channels.append(rec_channel)

        merged_path = os.path.join(output_dir, f"{base_name}_reconstruction_merged.png")
        cv2.imwrite(merged_path, cv2.merge(reconstructed_channels))

        merged_multilayer_paths = []
        if Hyperparams.MUL_SAVE:
            for depth_index in range(OPTICS_CONFIG["layer_num"]):
                layer_channels = []
                for channel_name in CHANNEL_NAMES:
                    layer_path = os.path.join(
                        output_dir,
                        f"{base_name}_{channel_name}_multilayer_reconstruction{depth_index}.png",
                    )
                    layer_channel = cv2.imread(layer_path, cv2.IMREAD_GRAYSCALE)
                    if layer_channel is None:
                        raise FileNotFoundError(
                            f"Multilayer reconstruction channel not found: {layer_path}"
                        )
                    layer_channels.append(layer_channel)

                merged_layer_path = os.path.join(
                    output_dir,
                    f"{base_name}_multilayer_reconstruction{depth_index}_merged.png",
                )
                cv2.imwrite(merged_layer_path, cv2.merge(layer_channels))
                merged_multilayer_paths.append(merged_layer_path)

        psnr_values = [row["psnr"] for row in channel_results if row["psnr"] is not None]
        ssim_values = [row["ssim"] for row in channel_results if row["ssim"] is not None]
        avg_psnr = float(np.mean(psnr_values)) if psnr_values else None
        avg_ssim = float(np.mean(ssim_values)) if ssim_values else None

        print(
            f"Merged RGB reconstruction: {merged_path} "
            f"(avg PSNR={format_metric(avg_psnr, ' dB')}, avg SSIM={format_metric(avg_ssim)})"
        )
        if VERBOSE and merged_multilayer_paths:
            print(f"Merged multilayer reconstructions: {len(merged_multilayer_paths)}")
        return True, avg_psnr, avg_ssim
    except Exception as exc:
        print(f"RGB merged reconstruction failed: {exc}")
        if VERBOSE:
            import traceback

            traceback.print_exc()
        return False, None, None
    finally:
        OPTICS_CONFIG["channel"] = original_channel


def find_image_pairs(input_dir):
    image_pairs = []
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".png") or name.endswith("_depth.png"):
            continue
        rgb_path = os.path.join(input_dir, name)
        depth_path = os.path.join(input_dir, name.replace(".png", "_depth.png"))
        if os.path.exists(depth_path):
            image_pairs.append((rgb_path, depth_path))
    return image_pairs


def predict_batch(
    model_path,
    input_dir,
    output_dir,
    use_structural_propagation=False,
    save_asm_input=False,
    colorize_asm_input_phase=False,
    merge_rgb_reconstruction=False,
):
    image_pairs = find_image_pairs(input_dir)
    if not image_pairs:
        print(f"No valid image pairs found in {input_dir}")
        print("Expected files: image.png and image_depth.png")
        return 0, 0, {}

    print(f"Found {len(image_pairs)} image pairs")
    success_count = 0
    psnr_list = []
    ssim_list = []
    results_summary = {}

    for index, (rgb_path, depth_path) in enumerate(image_pairs, start=1):
        print(f"[{index}/{len(image_pairs)}] {Path(rgb_path).name}")
        if merge_rgb_reconstruction:
            success, psnr, ssim = predict_rgb_merged_image(
                model_path,
                rgb_path,
                depth_path,
                output_dir,
                use_structural_propagation,
                save_asm_input,
                colorize_asm_input_phase,
            )
        else:
            success, psnr, ssim = predict_single_image(
                model_path,
                rgb_path,
                depth_path,
                output_dir,
                use_structural_propagation,
                save_asm_input,
                colorize_asm_input_phase,
            )

        file_name = Path(rgb_path).name
        results_summary[file_name] = {"success": success, "psnr": psnr, "ssim": ssim}
        if success:
            success_count += 1
            if psnr is not None:
                psnr_list.append(psnr)
            if ssim is not None:
                ssim_list.append(ssim)

    avg_psnr = float(np.mean(psnr_list)) if psnr_list else None
    avg_ssim = float(np.mean(ssim_list)) if ssim_list else None
    save_batch_summary(output_dir, image_pairs, success_count, avg_psnr, avg_ssim, results_summary)
    print(f"Batch complete: {success_count}/{len(image_pairs)} successful")
    return success_count, len(image_pairs), results_summary


def save_batch_summary(output_dir, image_pairs, success_count, avg_psnr, avg_ssim, results_summary):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "batch_prediction_summary.txt")
    lines = [
        "Batch Prediction Summary",
        "========================",
        f"Total processed: {len(image_pairs)}",
        f"Successful: {success_count}",
        f"Success rate: {success_count / len(image_pairs) * 100:.1f}%",
        "",
        "Quality Metrics",
        "---------------",
        f"Average PSNR: {format_metric(avg_psnr, ' dB')}",
        f"Average SSIM: {format_metric(avg_ssim)}",
        "",
        "Detailed Results",
        "----------------",
    ]
    for file_name, result in results_summary.items():
        if result["success"] and result["psnr"] is not None:
            lines.append(
                f"OK {file_name} - PSNR: {result['psnr']:.4f}, SSIM: {result['ssim']:.4f}"
            )
        else:
            lines.append(f"FAILED {file_name}")

    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if VERBOSE:
        print(f"Batch summary: {summary_path}")


def list_output_files(output_dir):
    if not VERBOSE:
        return
    if not os.path.exists(output_dir):
        print(f"Output directory does not exist: {output_dir}")
        return

    files = sorted(os.listdir(output_dir))
    if not files:
        print(f"Output directory is empty: {output_dir}")
        return

    total_size = 0
    print(f"Output files in {output_dir}:")
    for name in files:
        file_path = os.path.join(output_dir, name)
        file_size = os.path.getsize(file_path)
        total_size += file_size
        size = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
        print(f"  {name} ({size})")
    print(f"Total output size: {total_size / (1024 * 1024):.1f} MB")


def main():
    model_path = resolve_repo_path(MODEL_PATH)
    output_dir = resolve_repo_path(OUTPUT_DIR)
    device = configure_device(GPU_ID)
    print_configuration(model_path, output_dir, device)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if BATCH_MODE:
        input_dir = resolve_repo_path(INPUT_DIR)
        if not input_dir.is_dir():
            print(f"Input directory does not exist: {input_dir}")
            return
        predict_batch(
            str(model_path),
            str(input_dir),
            str(output_dir),
            USE_STRUCTURAL_PROPAGATION,
            SAVE_ASM_INPUT,
            COLORIZE_ASM_INPUT_PHASE,
            merge_rgb_reconstruction=MERGE_RGB_RECONSTRUCTION,
        )
    else:
        input_image = resolve_repo_path(INPUT_IMAGE)
        input_depth = resolve_repo_path(INPUT_DEPTH)
        if not input_image.exists() or not input_depth.exists():
            print("Input image or depth map does not exist")
            print(f"  image: {input_image}")
            print(f"  depth: {input_depth}")
            print("Expected example layout:")
            print("  data/example_input/image.png")
            print("  data/example_input/image_depth.png")
            return

        if MERGE_RGB_RECONSTRUCTION:
            predict_rgb_merged_image(
                str(model_path),
                str(input_image),
                str(input_depth),
                str(output_dir),
                USE_STRUCTURAL_PROPAGATION,
                SAVE_ASM_INPUT,
                COLORIZE_ASM_INPUT_PHASE,
            )
        else:
            predict_single_image(
                str(model_path),
                str(input_image),
                str(input_depth),
                str(output_dir),
                USE_STRUCTURAL_PROPAGATION,
                SAVE_ASM_INPUT,
                COLORIZE_ASM_INPUT_PHASE,
            )

    list_output_files(str(output_dir))


if __name__ == "__main__":
    main()
