"""Reconstruct one image from an existing phase hologram."""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Optics import Optics
from Trainer import Trainer


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def reconstruct_single(
    rgb_path,
    depth_path,
    hologram_path,
    output_path,
    prop_dist,
    wavelength,
    feature_size,
    channel,
    layer_num,
    layer_delta,
    phase_grating=True,
):
    """Run numerical reconstruction for one existing 8-bit phase hologram."""
    for label, path in {
        "RGB image": rgb_path,
        "depth map": depth_path,
        "phase hologram": hologram_path,
    }.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    eval_optics = Optics(
        prop_dist=prop_dist,
        wavelength=wavelength,
        feature_size=feature_size,
        channel=channel,
        layer_num=layer_num,
        layer_delta=layer_delta,
    )
    trainer = Trainer(
        train_optics_ins=None,
        eval_optics_ins=eval_optics,
        phs_code_model=None,
        root_path=os.path.dirname(hologram_path),
    )

    print(f"Reconstructing {hologram_path}")
    trainer.phs_predict(
        rgb_path=rgb_path,
        depth_path=depth_path,
        phs_path=hologram_path,
        img_rec_path=output_path,
        phase_grating=phase_grating,
    )
    print(f"Saved reconstruction: {output_path}")


if __name__ == "__main__":
    RGB_PATH = resolve_repo_path("data/example_input/bbb_rgb.png")
    DEPTH_PATH = resolve_repo_path("data/example_input/bbb_depth.png")
    HOLOGRAM_PATH = resolve_repo_path("data/example_holograms/green.png")
    OUTPUT_PATH = resolve_repo_path("outputs/single_reconstruction.png")

    reconstruct_single(
        rgb_path=str(RGB_PATH),
        depth_path=str(DEPTH_PATH),
        hologram_path=str(HOLOGRAM_PATH),
        output_path=str(OUTPUT_PATH),
        prop_dist=0.0001,
        wavelength=520e-9,
        feature_size=8e-6,
        channel=1,
        layer_num=3,
        layer_delta=0.0001,
        phase_grating=False,
    )
