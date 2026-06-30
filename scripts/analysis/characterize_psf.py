"""
PSF characterization for SGWN-Encoder under structural-only propagation.

This script runs two experiments:
  Part A: lateral PSF at multiple 2D propagation distances.
  Part B: axial depth selectivity for multiple points encoded in one hologram.

All backward propagation, network-input generation, and forward reconstruction use
only the structural ASM term. The global propagation phase is explicitly set to 1.
"""

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import csv
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Optics import Optics
from hyperparams import Hyperparams
from checkpoint_utils import load_phase_model


STRUCTURAL_GLOBAL_PHASE = None
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class PointSpec:
    name: str
    x: int
    y: int
    z_mm: float


def parse_float_list(value: str) -> List[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(SCRIPT_DIR, path)


def select_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available; falling back to CPU.")
        return torch.device("cpu")
    device = torch.device(device_name)
    Hyperparams.device = device
    return device


def structural_global(device: torch.device) -> torch.Tensor:
    global STRUCTURAL_GLOBAL_PHASE
    if STRUCTURAL_GLOBAL_PHASE is None or STRUCTURAL_GLOBAL_PHASE.device != device:
        STRUCTURAL_GLOBAL_PHASE = torch.tensor(1.0, dtype=torch.complex64, device=device)
    return STRUCTURAL_GLOBAL_PHASE


def make_optics(args: argparse.Namespace, z_mm: float) -> Optics:
    z_m = z_mm * 1e-3
    return Optics(
        channel=args.channel,
        first_z=z_m,
        delta_z=z_m,
        layer_num=1,
        factor=args.factor,
        grating_type=args.grating_type,
        LCoS_res_h=args.height,
        LCoS_res_w=args.width,
        LCoS_pitch=args.pitch,
        linear_conv=not args.no_linear_conv,
        band_limit=not args.no_band_limit,
    )


def load_model(model_path, device):
    return load_phase_model(model_path, device=device)


def draw_point_field(
    height: int,
    width: int,
    x: int,
    y: int,
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    amp = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius**2
    amp[0, 0][mask] = 1.0
    return torch.polar(amp, torch.zeros_like(amp))


def normalize_complex_field(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return u / (torch.max(torch.abs(u)) + eps)


def encode_phase(
    model,
    u_in: torch.Tensor,
    optics_for_grating: Optics,
    apply_phase_grating: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        raw_phase = model(normalize_complex_field(u_in))
        phase_for_prop = raw_phase
        if apply_phase_grating:
            phase_for_prop = phase_for_prop - optics_for_grating.phase_grating
    return raw_phase, phase_for_prop


def structural_backward(optics: Optics, u_target: torch.Tensor, device: torch.device) -> torch.Tensor:
    return optics.prop_asm(optics.h_backward_s, structural_global(device), u0=u_target)


def structural_forward_from_phase(
    optics: Optics,
    phase: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    return optics.prop_asm(optics.h_forward_s, structural_global(device), phs_in=phase)


def phase_to_uint8(phase: torch.Tensor) -> np.ndarray:
    phase_np = phase.detach().squeeze().cpu().numpy()
    phase_norm = ((phase_np - phase_np.mean() + np.pi) % (2 * np.pi)) / (2 * np.pi)
    return np.clip(np.round(phase_norm * 255), 0, 255).astype(np.uint8)


def intensity_to_uint8(intensity: np.ndarray, percentile: float = 99.9) -> np.ndarray:
    scale = np.percentile(intensity, percentile)
    if scale <= 0:
        scale = float(np.max(intensity))
    if scale <= 0:
        return np.zeros_like(intensity, dtype=np.uint8)
    return np.clip(np.round(intensity / scale * 255), 0, 255).astype(np.uint8)


def intensity_to_uint8_maxscale(intensity: np.ndarray) -> np.ndarray:
    scale = float(np.max(intensity))
    if scale <= 0:
        return np.zeros_like(intensity, dtype=np.uint8)
    return np.clip(np.round(intensity / scale * 255), 0, 255).astype(np.uint8)


def intensity_to_uint8_logscale(intensity: np.ndarray) -> np.ndarray:
    scale = float(np.max(intensity))
    if scale <= 0:
        return np.zeros_like(intensity, dtype=np.uint8)
    normalized = np.clip(intensity / scale, 0, None)
    log_img = np.log1p(1000.0 * normalized) / np.log1p(1000.0)
    return np.clip(np.round(log_img * 255), 0, 255).astype(np.uint8)


def crop_around(image: np.ndarray, x: int, y: int, half_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = max(0, x - half_size)
    x1 = min(w, x + half_size + 1)
    y0 = max(0, y - half_size)
    y1 = min(h, y + half_size + 1)
    return image[y0:y1, x0:x1]


def save_image_triplet(
    out_dir: str,
    prefix: str,
    target_u: torch.Tensor,
    raw_phase: torch.Tensor,
    intensity: np.ndarray,
    peak_x: int,
    peak_y: int,
    crop_half_size: int,
) -> None:
    target_amp = torch.abs(target_u).detach().squeeze().cpu().numpy()
    target_intensity = target_amp ** 2
    target_u8 = intensity_to_uint8(target_intensity, percentile=100.0)
    phase_u8 = phase_to_uint8(raw_phase)

    # Match Trainer.predict(): scale reconstructed intensity by total target energy.
    coefficient = float(np.sum(target_intensity) / (np.sum(intensity) + 1e-12))
    recon_predict_u8 = np.clip(np.round(intensity * coefficient * 255), 0, 255).astype(np.uint8)

    # Display-only views. Autoscale emphasizes background/sidelobes; maxscale preserves spot shape.
    recon_autoscale_u8 = intensity_to_uint8(intensity)
    recon_maxscale_u8 = intensity_to_uint8_maxscale(intensity)
    recon_logscale_u8 = intensity_to_uint8_logscale(intensity)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_target.png"), target_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_hologram.png"), phase_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction.png"), recon_predict_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_autoscale.png"), recon_autoscale_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_maxscale.png"), recon_maxscale_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_logscale.png"), recon_logscale_u8)
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_crop.png"),
        crop_around(recon_predict_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_autoscale_crop.png"),
        crop_around(recon_autoscale_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_maxscale_crop.png"),
        crop_around(recon_maxscale_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_logscale_crop.png"),
        crop_around(recon_logscale_u8, peak_x, peak_y, crop_half_size),
    )



def save_baseline_image_pair(
    out_dir: str,
    prefix: str,
    target_u: torch.Tensor,
    intensity: np.ndarray,
    peak_x: int,
    peak_y: int,
    crop_half_size: int,
) -> None:
    target_amp = torch.abs(target_u).detach().squeeze().cpu().numpy()
    target_intensity = target_amp ** 2
    coefficient = float(np.sum(target_intensity) / (np.sum(intensity) + 1e-12))
    recon_u8 = np.clip(np.round(intensity * coefficient * 255), 0, 255).astype(np.uint8)
    recon_autoscale_u8 = intensity_to_uint8(intensity)
    recon_maxscale_u8 = intensity_to_uint8_maxscale(intensity)
    recon_logscale_u8 = intensity_to_uint8_logscale(intensity)

    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction.png"), recon_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_autoscale.png"), recon_autoscale_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_maxscale.png"), recon_maxscale_u8)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_reconstruction_logscale.png"), recon_logscale_u8)
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_crop.png"),
        crop_around(recon_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_autoscale_crop.png"),
        crop_around(recon_autoscale_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_maxscale_crop.png"),
        crop_around(recon_maxscale_u8, peak_x, peak_y, crop_half_size),
    )
    cv2.imwrite(
        os.path.join(out_dir, f"{prefix}_reconstruction_logscale_crop.png"),
        crop_around(recon_logscale_u8, peak_x, peak_y, crop_half_size),
    )

def fwhm_1d(profile: np.ndarray) -> float:
    profile = np.asarray(profile, dtype=np.float64)
    max_val = float(np.max(profile))
    if not np.isfinite(max_val) or max_val <= 0:
        return float("nan")

    y = profile / max_val
    peak = int(np.argmax(y))
    half = 0.5

    left = peak
    while left > 0 and y[left] >= half:
        left -= 1
    if left == peak:
        left_cross = float(peak)
    elif y[left + 1] == y[left]:
        left_cross = float(left)
    else:
        left_cross = left + (half - y[left]) / (y[left + 1] - y[left])

    right = peak
    while right < len(y) - 1 and y[right] >= half:
        right += 1
    if right == peak:
        right_cross = float(peak)
    elif y[right - 1] == y[right]:
        right_cross = float(right)
    else:
        right_cross = (right - 1) + (half - y[right - 1]) / (y[right] - y[right - 1])

    return float(max(0.0, right_cross - left_cross))


def lateral_metrics(
    intensity: np.ndarray,
    target_x: int,
    target_y: int,
    pitch_m: float,
    background_exclusion: int,
) -> Dict[str, float]:
    peak_flat = int(np.argmax(intensity))
    peak_y, peak_x = np.unravel_index(peak_flat, intensity.shape)
    peak = float(intensity[peak_y, peak_x])

    fwhm_x_px = fwhm_1d(intensity[peak_y, :])
    fwhm_y_px = fwhm_1d(intensity[:, peak_x])
    fwhm_lat_px = float(np.nanmean([fwhm_x_px, fwhm_y_px]))

    mask = np.ones_like(intensity, dtype=bool)
    y0 = max(0, peak_y - background_exclusion)
    y1 = min(intensity.shape[0], peak_y + background_exclusion + 1)
    x0 = max(0, peak_x - background_exclusion)
    x1 = min(intensity.shape[1], peak_x + background_exclusion + 1)
    mask[y0:y1, x0:x1] = False
    bg_mean = float(np.mean(intensity[mask]))
    sbr_db = 10.0 * math.log10((peak + 1e-12) / (bg_mean + 1e-12))

    drift_px = math.hypot(float(peak_x - target_x), float(peak_y - target_y))

    return {
        "peak_x": float(peak_x),
        "peak_y": float(peak_y),
        "peak": peak,
        "background_mean": bg_mean,
        "fwhm_x_px": fwhm_x_px,
        "fwhm_y_px": fwhm_y_px,
        "fwhm_lat_px": fwhm_lat_px,
        "fwhm_x_um": fwhm_x_px * pitch_m * 1e6,
        "fwhm_y_um": fwhm_y_px * pitch_m * 1e6,
        "fwhm_lat_um": fwhm_lat_px * pitch_m * 1e6,
        "sbr_db": sbr_db,
        "drift_px": drift_px,
        "drift_um": drift_px * pitch_m * 1e6,
    }


def save_profile_plot(
    out_path: str,
    intensity: np.ndarray,
    peak_x: int,
    peak_y: int,
    title: str,
) -> None:
    profile_x = intensity[peak_y, :]
    profile_y = intensity[:, peak_x]
    profile_x = profile_x / (np.max(profile_x) + 1e-12)
    profile_y = profile_y / (np.max(profile_y) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(profile_x)
    axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[0].set_title("Horizontal profile")
    axes[0].set_xlabel("x pixel")
    axes[0].set_ylabel("Normalized intensity")

    axes[1].plot(profile_y)
    axes[1].axhline(0.5, color="k", linestyle="--", linewidth=1)
    axes[1].set_title("Vertical profile")
    axes[1].set_xlabel("y pixel")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_csv(path: str, rows: Sequence[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_part_a_summary(rows: Sequence[Dict[str, float]], args: argparse.Namespace, out_path: str) -> None:
    z_mm = np.array([r["z_mm"] for r in rows], dtype=np.float64)
    fwhm_x_um = np.array([r["fwhm_x_um"] for r in rows], dtype=np.float64)
    fwhm_y_um = np.array([r["fwhm_y_um"] for r in rows], dtype=np.float64)
    fwhm_lat_um = np.array([r["fwhm_lat_um"] for r in rows], dtype=np.float64)
    sbr_db = np.array([r["sbr_db"] for r in rows], dtype=np.float64)

    wavelength = {0: 4.5e-7, 1: 5.2e-7, 2: 6.38e-7}[args.channel]
    z_m = z_mm * 1e-3
    dx = args.width * args.pitch
    dy = args.height * args.pitch
    theory_x_um = 0.886 * wavelength * z_m / dx * 1e6
    theory_y_um = 0.886 * wavelength * z_m / dy * 1e6
    theory_lat_um = 0.5 * (theory_x_um + theory_y_um)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(z_mm, fwhm_x_um, "o-", label="FWHM x")
    axes[0].plot(z_mm, fwhm_y_um, "s-", label="FWHM y")
    axes[0].plot(z_mm, fwhm_lat_um, "^-", label="FWHM avg")
    axes[0].plot(z_mm, theory_x_um, "--", label="0.886 lambda z / Dx")
    axes[0].plot(z_mm, theory_y_um, "--", label="0.886 lambda z / Dy")
    axes[0].plot(z_mm, theory_lat_um, "k:", label="Theory avg")
    axes[0].set_xlabel("Propagation distance z (mm)")
    axes[0].set_ylabel("FWHM (um)")
    axes[0].set_title("Lateral PSF width")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].plot(z_mm, sbr_db, "o-", color="tab:green")
    axes[1].set_xlabel("Propagation distance z (mm)")
    axes[1].set_ylabel("SBR (dB)")
    axes[1].set_title("Signal-to-background ratio")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_part_a_method_comparison(method_rows: Dict[str, Sequence[Dict[str, float]]], args: argparse.Namespace, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, rows in method_rows.items():
        z_mm = np.array([r["z_mm"] for r in rows], dtype=np.float64)
        fwhm_lat_um = np.array([r["fwhm_lat_um"] for r in rows], dtype=np.float64)
        sbr_db = np.array([r["sbr_db"] for r in rows], dtype=np.float64)
        axes[0].plot(z_mm, fwhm_lat_um, "o-", label=method)
        axes[1].plot(z_mm, sbr_db, "o-", label=method)

    axes[0].set_xlabel("Propagation distance z (mm)")
    axes[0].set_ylabel("Lateral FWHM (um)")
    axes[0].set_title("Part A method comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel("Propagation distance z (mm)")
    axes[1].set_ylabel("SBR (dB)")
    axes[1].set_title("SBR comparison")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def run_part_a(model, args: argparse.Namespace, device: torch.device, out_root: str) -> List[Dict[str, float]]:
    out_dir = os.path.join(out_root, "part_a_lateral_psf")
    ensure_dir(out_dir)

    target_x = args.width // 2
    target_y = args.height // 2
    rows_sgwn: List[Dict[str, float]] = []
    rows_structural: List[Dict[str, float]] = []
    rows_traditional: List[Dict[str, float]] = []

    for z_mm in args.part_a_distances:
        print(f"[Part A] z = {z_mm:.3f} mm")
        optics = make_optics(args, z_mm)
        target_u = draw_point_field(args.height, args.width, target_x, target_y, args.point_radius, device)

        with torch.no_grad():
            u_in = structural_backward(optics, target_u, device)

            raw_phase, phase_for_prop = encode_phase(
                model, u_in, optics, apply_phase_grating=not args.no_phase_grating
            )
            rec_u_sgwn = structural_forward_from_phase(optics, phase_for_prop, device)
            intensity_sgwn = (torch.abs(rec_u_sgwn) ** 2).detach().squeeze().cpu().numpy()

            rec_u_structural = optics.prop_asm(optics.h_forward_s, structural_global(device), u0=u_in)
            intensity_structural = (torch.abs(rec_u_structural) ** 2).detach().squeeze().cpu().numpy()

            rec_u_traditional = optics.prop_asm(optics.h_forward_s, optics.h_forward_g, u0=u_in)
            intensity_traditional = (torch.abs(rec_u_traditional) ** 2).detach().squeeze().cpu().numpy()

        prefix = f"z_{z_mm:g}mm".replace(".", "p")

        metrics_sgwn = lateral_metrics(
            intensity_sgwn, target_x, target_y, args.pitch, background_exclusion=args.background_exclusion
        )
        rows_sgwn.append({"z_mm": float(z_mm), **metrics_sgwn})
        save_image_triplet(
            out_dir, prefix, target_u, raw_phase, intensity_sgwn,
            int(metrics_sgwn["peak_x"]), int(metrics_sgwn["peak_y"]), args.crop_half_size,
        )
        save_profile_plot(
            os.path.join(out_dir, f"{prefix}_profiles.png"),
            intensity_sgwn, int(metrics_sgwn["peak_x"]), int(metrics_sgwn["peak_y"]),
            f"SGWN-Encoder structural-only lateral PSF, z={z_mm:g} mm",
        )

        metrics_structural = lateral_metrics(
            intensity_structural, target_x, target_y, args.pitch, background_exclusion=args.background_exclusion
        )
        rows_structural.append({"z_mm": float(z_mm), **metrics_structural})
        save_baseline_image_pair(
            out_dir, f"{prefix}_baseline_structural", target_u, intensity_structural,
            int(metrics_structural["peak_x"]), int(metrics_structural["peak_y"]), args.crop_half_size,
        )
        save_profile_plot(
            os.path.join(out_dir, f"{prefix}_baseline_structural_profiles.png"),
            intensity_structural, int(metrics_structural["peak_x"]), int(metrics_structural["peak_y"]),
            f"Structural ASM baseline, z={z_mm:g} mm",
        )

        metrics_traditional = lateral_metrics(
            intensity_traditional, target_x, target_y, args.pitch, background_exclusion=args.background_exclusion
        )
        rows_traditional.append({"z_mm": float(z_mm), **metrics_traditional})
        save_baseline_image_pair(
            out_dir, f"{prefix}_baseline_traditional", target_u, intensity_traditional,
            int(metrics_traditional["peak_x"]), int(metrics_traditional["peak_y"]), args.crop_half_size,
        )
        save_profile_plot(
            os.path.join(out_dir, f"{prefix}_baseline_traditional_profiles.png"),
            intensity_traditional, int(metrics_traditional["peak_x"]), int(metrics_traditional["peak_y"]),
            f"Traditional ASM baseline, z={z_mm:g} mm",
        )

        del optics, target_u, u_in, raw_phase, phase_for_prop
        del rec_u_sgwn, rec_u_structural, rec_u_traditional
        if device.type == "cuda":
            torch.cuda.empty_cache()

    write_csv(os.path.join(out_dir, "part_a_metrics.csv"), rows_sgwn)
    write_csv(os.path.join(out_dir, "part_a_baseline_structural_metrics.csv"), rows_structural)
    write_csv(os.path.join(out_dir, "part_a_baseline_traditional_metrics.csv"), rows_traditional)
    plot_part_a_summary(rows_sgwn, args, os.path.join(out_dir, "part_a_summary.png"))
    plot_part_a_method_comparison(
        {
            "SGWN-Encoder": rows_sgwn,
            "Structural ASM": rows_structural,
            "Traditional ASM": rows_traditional,
        },
        args,
        os.path.join(out_dir, "part_a_method_comparison.png"),
    )
    return rows_sgwn

def default_part_b_points(width: int, height: int, z_values: Sequence[float]) -> List[PointSpec]:
    locations = [(0.30, 0.45), (0.50, 0.55), (0.70, 0.45)]
    names = ["A", "B", "C"]
    return [
        PointSpec(names[i], int(round(width * locations[i][0])), int(round(height * locations[i][1])), z_values[i])
        for i in range(len(z_values))
    ]


def dense_grid_part_b_points(depths: Sequence[float], xs: Sequence[float], ys: Sequence[float]) -> List[PointSpec]:
    if len(depths) != len(ys):
        raise ValueError("Dense Part B requires the same number of depths and y rows.")
    points: List[PointSpec] = []
    for row_idx, (z_mm, y) in enumerate(zip(depths, ys), start=1):
        for col_idx, x in enumerate(xs, start=1):
            points.append(PointSpec(f"R{row_idx:02d}C{col_idx:02d}", int(round(x)), int(round(y)), float(z_mm)))
    return points


def unique_sorted(values: Sequence[float]) -> np.ndarray:
    return np.array(sorted({float(v) for v in values}), dtype=np.float64)


def window_max(intensity: np.ndarray, x: int, y: int, window: int) -> float:
    half = window // 2
    patch = crop_around(intensity, x, y, half)
    return float(np.max(patch))


def fwhm_axis(z_mm: np.ndarray, values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    max_val = float(np.max(values))
    if not np.isfinite(max_val) or max_val <= 0:
        return float("nan")
    y = values / max_val
    peak = int(np.argmax(y))
    half = 0.5

    left = peak
    while left > 0 and y[left] >= half:
        left -= 1
    if left == peak:
        left_cross = float(z_mm[peak])
    else:
        denom = y[left + 1] - y[left]
        frac = 0.0 if denom == 0 else (half - y[left]) / denom
        left_cross = float(z_mm[left] + frac * (z_mm[left + 1] - z_mm[left]))

    right = peak
    while right < len(y) - 1 and y[right] >= half:
        right += 1
    if right == peak:
        right_cross = float(z_mm[peak])
    else:
        denom = y[right] - y[right - 1]
        frac = 0.0 if denom == 0 else (half - y[right - 1]) / denom
        right_cross = float(z_mm[right - 1] + frac * (z_mm[right] - z_mm[right - 1]))

    return max(0.0, right_cross - left_cross)


def plot_part_b_response(
    scan_z: np.ndarray,
    responses: Dict[str, np.ndarray],
    points: Sequence[PointSpec],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(points) > 10:
        depths = unique_sorted([p.z_mm for p in points])
        for z_mm in depths:
            group = [responses[p.name] / (np.max(responses[p.name]) + 1e-12) for p in points if p.z_mm == z_mm]
            values = np.stack(group, axis=0)
            mean = np.mean(values, axis=0)
            std = np.std(values, axis=0)
            ax.plot(scan_z, mean, label=f"{z_mm:g} mm")
            ax.fill_between(scan_z, np.maximum(0, mean - std), np.minimum(1, mean + std), alpha=0.12)
            ax.axvline(z_mm, color="k", linestyle="--", linewidth=0.6, alpha=0.2)
    else:
        for p in points:
            values = responses[p.name]
            values = values / (np.max(values) + 1e-12)
            ax.plot(scan_z, values, label=f"Point {p.name}, target {p.z_mm:g} mm")
            ax.axvline(p.z_mm, color="k", linestyle="--", linewidth=0.8, alpha=0.35)
    ax.set_xlabel("Reconstruction distance z (mm)")
    ax.set_ylabel("Normalized local peak intensity")
    ax.set_title("Axial response under structural-only propagation")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2 if len(points) > 10 else 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def plot_crosstalk_matrix(
    matrix: np.ndarray,
    points: Sequence[PointSpec],
    out_path: str,
    column_depths: Sequence[float] | None = None,
) -> None:
    row_norm = matrix / (np.max(matrix, axis=1, keepdims=True) + 1e-12)
    if column_depths is None:
        column_depths = [p.z_mm for p in points]
    fig_h = max(4.0, 0.24 * len(points) + 1.5)
    fig_w = max(5.0, 0.42 * len(column_depths) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(row_norm, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(column_depths)))
    ax.set_xticklabels([f"{z:g} mm" for z in column_depths], rotation=45, ha="right")
    ax.set_yticks(range(len(points)))
    ax.set_yticklabels([f"{p.name} ({p.z_mm:g})" for p in points], fontsize=7 if len(points) > 10 else 9)
    ax.set_xlabel("Reconstruction depth")
    ax.set_ylabel("Point location and target depth")
    ax.set_title("Row-normalized depth cross-talk")
    if row_norm.size <= 80:
        for i in range(row_norm.shape[0]):
            for j in range(row_norm.shape[1]):
                ax.text(j, i, f"{row_norm[i, j]:.2f}", ha="center", va="center", color="w", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def run_part_b(model, args: argparse.Namespace, device: torch.device, out_root: str) -> List[Dict[str, float]]:
    dense_mode = args.part_b_layout == "dense-grid"
    out_dir = os.path.join(out_root, "part_b_dense_depth_grid" if dense_mode else "part_b_axial_psf")
    ensure_dir(out_dir)

    if dense_mode:
        points = dense_grid_part_b_points(args.part_b_depths, args.part_b_grid_xs, args.part_b_grid_ys)
        analysis_depths = unique_sorted(args.part_b_depths)
        display_depths = np.array(args.part_b_display_depths, dtype=np.float64)
    else:
        points = default_part_b_points(args.width, args.height, args.part_b_depths)
        analysis_depths = unique_sorted([p.z_mm for p in points])
        display_depths = analysis_depths

    with open(os.path.join(out_dir, "point_specs.json"), "w") as f:
        json.dump([p.__dict__ for p in points], f, indent=2)
    with open(os.path.join(out_dir, "part_b_config.json"), "w") as f:
        json.dump(
            {
                "layout": args.part_b_layout,
                "analysis_depths_mm": analysis_depths.tolist(),
                "display_depths_mm": display_depths.tolist(),
                "scan_start_mm": args.part_b_scan_start,
                "scan_stop_mm": args.part_b_scan_stop,
                "scan_step_mm": args.part_b_scan_step,
            },
            f,
            indent=2,
        )

    u_in = torch.zeros((1, 1, args.height, args.width), dtype=torch.complex64, device=device)
    target_sum = torch.zeros_like(u_in)
    target_intensity_sum = torch.zeros((1, 1, args.height, args.width), dtype=torch.float32, device=device)
    grating_optics = None

    print("[Part B] Building multi-depth structural-only input field")
    for p in points:
        print(f"  point {p.name}: x={p.x}, y={p.y}, z={p.z_mm:g} mm")
        optics = make_optics(args, p.z_mm)
        if grating_optics is None:
            grating_optics = optics
        target_u = draw_point_field(args.height, args.width, p.x, p.y, args.point_radius, device)
        with torch.no_grad():
            u_in = u_in + structural_backward(optics, target_u, device)
            target_sum = target_sum + target_u
            target_intensity_sum = target_intensity_sum + torch.abs(target_u) ** 2
        if optics is not grating_optics:
            del optics
        del target_u
        if device.type == "cuda":
            torch.cuda.empty_cache()

    u_in = normalize_complex_field(u_in)
    raw_phase, phase_for_prop = encode_phase(
        model, u_in, grating_optics, apply_phase_grating=not args.no_phase_grating
    )

    cv2.imwrite(os.path.join(out_dir, "multi_depth_target.png"), intensity_to_uint8(torch.abs(target_sum).squeeze().cpu().numpy(), 100.0))
    cv2.imwrite(os.path.join(out_dir, "multi_depth_hologram.png"), phase_to_uint8(raw_phase))

    scan_z = np.arange(args.part_b_scan_start, args.part_b_scan_stop + args.part_b_scan_step * 0.5, args.part_b_scan_step)
    responses = {p.name: np.zeros_like(scan_z, dtype=np.float64) for p in points}
    saved_plane_intensities: Dict[float, np.ndarray] = {}
    depths_to_cache = unique_sorted(list(analysis_depths) + list(display_depths))

    print(f"[Part B] Axial scan: {args.part_b_scan_start:g}:{args.part_b_scan_step:g}:{args.part_b_scan_stop:g} mm")
    for idx, z_mm in enumerate(scan_z):
        if idx % max(1, int(round(5.0 / args.part_b_scan_step))) == 0:
            print(f"  scan z = {z_mm:.3f} mm ({idx + 1}/{len(scan_z)})")

        optics = make_optics(args, float(z_mm))
        with torch.no_grad():
            rec_u = structural_forward_from_phase(optics, phase_for_prop, device)
            intensity = (torch.abs(rec_u) ** 2).detach().squeeze().cpu().numpy()

        for p in points:
            responses[p.name][idx] = window_max(intensity, p.x, p.y, args.axial_window)

        for depth in depths_to_cache:
            if abs(float(depth) - float(z_mm)) <= args.part_b_scan_step * 0.51:
                saved_plane_intensities[float(depth)] = intensity.copy()

        del optics, rec_u
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rows: List[Dict[str, float]] = []
    matrix = np.zeros((len(points), len(analysis_depths)), dtype=np.float64)

    for j, z_target in enumerate(analysis_depths):
        if float(z_target) not in saved_plane_intensities:
            nearest_idx = int(np.argmin(np.abs(scan_z - z_target)))
            z_nearest = float(scan_z[nearest_idx])
            optics = make_optics(args, z_nearest)
            with torch.no_grad():
                rec_u = structural_forward_from_phase(optics, phase_for_prop, device)
                saved_plane_intensities[float(z_target)] = (torch.abs(rec_u) ** 2).detach().squeeze().cpu().numpy()
            del optics, rec_u
        intensity = saved_plane_intensities[float(z_target)]
        for i, p in enumerate(points):
            matrix[i, j] = window_max(intensity, p.x, p.y, args.axial_window)

    for i, p in enumerate(points):
        response = responses[p.name]
        normalized = response / (np.max(response) + 1e-12)
        peak_idx = int(np.argmax(response))
        z_peak = float(scan_z[peak_idx])
        exclude = np.abs(scan_z - p.z_mm) <= args.dcr_exclusion_mm
        bg = float(np.mean(response[~exclude])) if np.any(~exclude) else float(np.mean(response))
        target_idx = int(np.argmin(np.abs(scan_z - p.z_mm)))
        target_depth_idx = int(np.argmin(np.abs(analysis_depths - p.z_mm)))
        target_response = float(response[target_idx])
        adjacent_depth_indices = [
            j for j, z in enumerate(analysis_depths)
            if j != target_depth_idx and abs(float(z) - p.z_mm) <= args.part_b_adjacent_delta_mm + 1e-9
        ]
        adjacent_response = float(np.max(matrix[i, adjacent_depth_indices])) if adjacent_depth_indices else float("nan")
        adjacent_ratio = adjacent_response / (target_response + 1e-12) if np.isfinite(adjacent_response) else float("nan")
        adjacent_suppression_db = 10.0 * math.log10((target_response + 1e-12) / (adjacent_response + 1e-12)) if np.isfinite(adjacent_response) else float("nan")
        dcr_db = 10.0 * math.log10((target_response + 1e-12) / (bg + 1e-12))

        rows.append(
            {
                "point": p.name,
                "x": float(p.x),
                "y": float(p.y),
                "target_z_mm": float(p.z_mm),
                "z_peak_mm": z_peak,
                "depth_error_mm": abs(z_peak - p.z_mm),
                "axial_fwhm_mm": fwhm_axis(scan_z, normalized),
                "target_response": target_response,
                "peak_response": float(response[peak_idx]),
                "adjacent_crosstalk_ratio": adjacent_ratio,
                "adjacent_suppression_db": adjacent_suppression_db,
                "dcr_db": dcr_db,
            }
        )

    target_intensity_np = target_intensity_sum.detach().squeeze().cpu().numpy()
    for z_target in display_depths:
        if float(z_target) not in saved_plane_intensities:
            nearest_idx = int(np.argmin(np.abs(scan_z - z_target)))
            z_nearest = float(scan_z[nearest_idx])
            optics = make_optics(args, z_nearest)
            with torch.no_grad():
                rec_u = structural_forward_from_phase(optics, phase_for_prop, device)
                saved_plane_intensities[float(z_target)] = (torch.abs(rec_u) ** 2).detach().squeeze().cpu().numpy()
            del optics, rec_u

    display_intensities = [saved_plane_intensities[float(z)] for z in display_depths]
    global_max_scale = max(float(np.max(img)) for img in display_intensities)
    global_p999_scale = float(np.percentile(np.concatenate([img.reshape(-1) for img in display_intensities]), 99.9))
    if global_max_scale <= 0:
        global_max_scale = 1.0
    if global_p999_scale <= 0:
        global_p999_scale = global_max_scale
    display_scale_rows = [{"global_max_scale": global_max_scale, "global_p999_scale": global_p999_scale}]
    write_csv(os.path.join(out_dir, "part_b_display_scales.csv"), display_scale_rows)

    for z_target in display_depths:
        intensity = saved_plane_intensities[float(z_target)]
        coefficient = float(np.sum(target_intensity_np) / (np.sum(intensity) + 1e-12))
        recon_u8 = np.clip(np.round(intensity * coefficient * 255), 0, 255).astype(np.uint8)
        recon_autoscale_u8 = intensity_to_uint8(intensity)
        recon_maxscale_u8 = intensity_to_uint8_maxscale(intensity)
        recon_logscale_u8 = intensity_to_uint8_logscale(intensity)
        recon_globalmax_u8 = np.clip(np.round(intensity / global_max_scale * 255), 0, 255).astype(np.uint8)
        recon_globalp999_u8 = np.clip(np.round(intensity / global_p999_scale * 255), 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm.png"), recon_u8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_autoscale.png"), recon_autoscale_u8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_maxscale.png"), recon_maxscale_u8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_logscale.png"), recon_logscale_u8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_globalmax.png"), recon_globalmax_u8)
        cv2.imwrite(os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_globalp999.png"), recon_globalp999_u8)
        for p in points:
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_crop.png"),
                crop_around(recon_u8, p.x, p.y, args.crop_half_size),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_autoscale_crop.png"),
                crop_around(recon_autoscale_u8, p.x, p.y, args.crop_half_size),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_maxscale_crop.png"),
                crop_around(recon_maxscale_u8, p.x, p.y, args.crop_half_size),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_logscale_crop.png"),
                crop_around(recon_logscale_u8, p.x, p.y, args.crop_half_size),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_globalmax_crop.png"),
                crop_around(recon_globalmax_u8, p.x, p.y, args.crop_half_size),
            )
            cv2.imwrite(
                os.path.join(out_dir, f"reconstruction_z_{z_target:g}mm_point_{p.name}_globalp999_crop.png"),
                crop_around(recon_globalp999_u8, p.x, p.y, args.crop_half_size),
            )

    write_csv(os.path.join(out_dir, "part_b_axial_metrics.csv"), rows)
    np.savetxt(os.path.join(out_dir, "part_b_crosstalk_matrix_raw.csv"), matrix, delimiter=",")
    np.savetxt(
        os.path.join(out_dir, "part_b_crosstalk_matrix_row_normalized.csv"),
        matrix / (np.max(matrix, axis=1, keepdims=True) + 1e-12),
        delimiter=",",
    )

    summary_rows: List[Dict[str, float]] = []
    for z_mm in analysis_depths:
        group = [r for r in rows if abs(r["target_z_mm"] - float(z_mm)) < 1e-9]
        summary: Dict[str, float] = {"target_z_mm": float(z_mm), "n_points": float(len(group))}
        for key in ["depth_error_mm", "axial_fwhm_mm", "dcr_db", "adjacent_crosstalk_ratio", "adjacent_suppression_db"]:
            values = np.array([float(r[key]) for r in group], dtype=np.float64)
            summary[f"{key}_mean"] = float(np.nanmean(values))
            summary[f"{key}_std"] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    write_csv(os.path.join(out_dir, "part_b_depth_summary.csv"), summary_rows)

    response_rows = []
    for idx, z_mm in enumerate(scan_z):
        row = {"z_mm": float(z_mm)}
        for p in points:
            row[f"point_{p.name}_response"] = float(responses[p.name][idx])
            row[f"point_{p.name}_normalized"] = float(responses[p.name][idx] / (np.max(responses[p.name]) + 1e-12))
        response_rows.append(row)
    write_csv(os.path.join(out_dir, "part_b_axial_response_curves.csv"), response_rows)

    plot_part_b_response(scan_z, responses, points, os.path.join(out_dir, "part_b_axial_response.png"))
    plot_crosstalk_matrix(matrix, points, os.path.join(out_dir, "part_b_crosstalk_matrix.png"), analysis_depths)

    del raw_phase, phase_for_prop, u_in, target_sum, target_intensity_sum
    if grating_optics is not None:
        del grating_optics
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SGWN-Encoder PSF characterization under structural-only propagation.")
    parser.add_argument("--model-path", default="checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt")
    parser.add_argument("--output-dir", default="outputs/psf_characterization/structural_only")
    parser.add_argument("--part", choices=["a", "b", "all"], default="all")
    parser.add_argument("--device", default="cuda:3")

    parser.add_argument("--channel", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--no-phase-grating", action="store_true")

    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--background-exclusion", type=int, default=25)
    parser.add_argument("--crop-half-size", type=int, default=96)

    parser.add_argument("--part-a-distances", type=parse_float_list, default=parse_float_list("5,50,100,200,500"))

    parser.add_argument("--part-b-layout", choices=["three-point", "dense-grid"], default="three-point")
    parser.add_argument("--part-b-depths", type=parse_float_list, default=parse_float_list("10,30,50"))
    parser.add_argument("--part-b-grid-xs", type=parse_float_list, default=parse_float_list("960,1920,2880"))
    parser.add_argument("--part-b-grid-ys", type=parse_float_list, default=parse_float_list("300,520,740,960,1180,1400,1620,1840"))
    parser.add_argument("--part-b-display-depths", type=parse_float_list, default=parse_float_list("12,16,20"))
    parser.add_argument("--part-b-scan-start", type=float, default=5.0)
    parser.add_argument("--part-b-scan-stop", type=float, default=60.0)
    parser.add_argument("--part-b-scan-step", type=float, default=0.2)
    parser.add_argument("--part-b-adjacent-delta-mm", type=float, default=2.0)
    parser.add_argument("--axial-window", type=int, default=21)
    parser.add_argument("--dcr-exclusion-mm", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    args.model_path = resolve_path(args.model_path)
    args.output_dir = resolve_path(args.output_dir)

    if args.part_b_layout == "three-point" and len(args.part_b_depths) != 3:
        raise ValueError("Three-point Part B expects exactly three depths, e.g. --part-b-depths 10,30,50")
    if args.part_b_layout == "dense-grid" and len(args.part_b_depths) != len(args.part_b_grid_ys):
        raise ValueError("Dense-grid Part B expects one y row per depth.")

    device = select_device(args.device)
    ensure_dir(args.output_dir)

    with open(os.path.join(args.output_dir, "run_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print("Propagation mode: structural-only (global phase fixed to 1)")
    if args.no_phase_grating:
        print("Phase grating compensation: disabled")
    else:
        print("Phase grating compensation: enabled, matching the project prediction path")

    model = load_model(args.model_path, device)

    if args.part in ("a", "all"):
        rows_a = run_part_a(model, args, device, args.output_dir)
        print(f"Part A complete: {len(rows_a)} distances")

    if args.part in ("b", "all"):
        rows_b = run_part_b(model, args, device, args.output_dir)
        print(f"Part B complete: {len(rows_b)} point responses")


if __name__ == "__main__":
    main()
