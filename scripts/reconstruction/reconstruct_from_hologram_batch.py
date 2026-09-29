"""Batch reconstruction from existing phase holograms.

Expects three directories with matching file stems:
  rgb_dir/{name}.png       target RGB image
  depth_dir/{name}.png     depth map for the same image
  hologram_dir/{name}.png  phase hologram to reconstruct
Edit the paths in __main__ for your data layout.
"""

import glob
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


def reconstruct_batch(
    rgb_dir,
    depth_dir,
    hologram_dir,
    output_dir,
    channel,
    first_z,
    delta_z,
    layer_num,
    LCoS_pitch=3.6e-6,
    LCoS_res_h=2160,
    LCoS_res_w=3840,
    factor=0.75,
    grating_type="vertical",
    phase_grating=True,
):
    """Reconstruct all PNG phase holograms in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    eval_optics = Optics(
        channel=channel,
        first_z=first_z,
        delta_z=delta_z,
        layer_num=layer_num,
        factor=factor,
        grating_type=grating_type,
        LCoS_res_h=LCoS_res_h,
        LCoS_res_w=LCoS_res_w,
        LCoS_pitch=LCoS_pitch,
    )
    trainer = Trainer(
        train_optics_ins=None,
        eval_optics_ins=eval_optics,
        phs_code_model=None,
        root_path=hologram_dir,
    )

    hologram_files = sorted(glob.glob(os.path.join(hologram_dir, "*.png")))
    if not hologram_files:
        print(f"No hologram files found: {hologram_dir}")
        return

    success_count = 0
    skipped_count = 0
    print(f"Reconstructing {len(hologram_files)} hologram(s)")
    for index, phs_path in enumerate(hologram_files, start=1):
        base_name = Path(phs_path).stem
        rgb_path = os.path.join(rgb_dir, f"{base_name}.png")
        depth_path = os.path.join(depth_dir, f"{base_name}.png")
        rec_path = os.path.join(output_dir, f"{base_name}_reconstruction.png")

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            skipped_count += 1
            print(f"[{index}/{len(hologram_files)}] skipped {base_name}: missing RGB or depth file")
            continue

        try:
            trainer.phs_predict(
                rgb_path=rgb_path,
                depth_path=depth_path,
                phs_path=phs_path,
                img_rec_path=rec_path,
                phase_grating=phase_grating,
            )
            success_count += 1
        except Exception as exc:
            skipped_count += 1
            print(f"[{index}/{len(hologram_files)}] failed {base_name}: {exc}")

    print(f"Batch reconstruction complete: {success_count} saved, {skipped_count} skipped")


if __name__ == "__main__":
    reconstruct_batch(
        rgb_dir=str(resolve_repo_path("data/rgb")),
        depth_dir=str(resolve_repo_path("data/depth")),
        hologram_dir=str(resolve_repo_path("holograms")),
        output_dir=str(resolve_repo_path("outputs/reconstructions")),
        channel=1,
        first_z=0.2,
        delta_z=0.04,
        layer_num=5,
        LCoS_pitch=8e-6,
        LCoS_res_h=2160,
        LCoS_res_w=3840,
        factor=0.75,
        grating_type="vertical",
        phase_grating=True,
    )
