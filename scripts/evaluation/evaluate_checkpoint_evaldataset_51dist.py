#!/usr/bin/env python3
"""
Evaluate one SGWN-Encoder checkpoint on evaldataset_rgb at 4K over the seed-8 51-distance list.

Defaults:
- model: checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt
- dataset: data/evaldataset_rgb
- size: 4k:3840x2160
- distances: [0.005, 0.1, 0.2] + 48 np.random.seed(8) samples from [0, 0.5]
- device: cuda:0

This is a thin experiment wrapper around evaluate_checkpoint_key_distances.py.
By default it saves hologram/reconstruction PNGs and records CSV metrics.
"""

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import evaluate_checkpoint_key_distances as evaluator


DEFAULT_MODEL_PATH = Path("checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt")
DEFAULT_INPUT_DIR = Path("data/evaldataset_rgb")
DEFAULT_OUTPUT_DIR = Path("outputs/checkpoint_evaldataset_51dist")


def build_distance_list():
    manual_distances = [0.005, 0.1, 0.2]
    np.random.seed(8)
    random_distances = np.random.uniform(low=0.0, high=0.5, size=48).tolist()
    return sorted(list(set(manual_distances + random_distances)))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one SGWN-Encoder checkpoint on evaldataset_rgb at 4K over the seed-8 51-distance list"
    )
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sizes", nargs="+", default=["4k:3840x2160"])
    parser.add_argument("--distances", type=float, nargs="+", default=None,
                        help="Override the default seed-8 51-distance list.")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--pitch", type=float, default=3.6e-6)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--factor", type=float, default=0.75)
    parser.add_argument("--grating-type", default="vertical")
    parser.add_argument("--no-linear-conv", action="store_true")
    parser.add_argument("--no-band-limit", action="store_true")
    parser.add_argument("--use-structural-propagation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Save hologram/reconstruction PNGs for every image. Use --no-save-images for metrics only.")
    args = parser.parse_args()
    if args.distances is None:
        args.distances = build_distance_list()
    return args


if __name__ == "__main__":
    evaluator.run(parse_args())
