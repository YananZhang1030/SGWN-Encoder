#!/usr/bin/env python3
"""Train SGWN-Encoder with dataset-free random complex fields."""

import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from CNNs import TPN_R
from Optics import Optics
from Trainer import Trainer
from checkpoint_utils import load_phase_model
from hyperparams import Hyperparams

# Training configuration
SEED = 42
GPU_ID = 0
DATASET_FREE_CONFIGS = ["DF_D_R_90_90_L_350_350"]
DATA_ROOT = "data"
EVAL_PATH = "eval"
OUTPUT_ROOT = "outputs/train_dataset_free"
RESUME_IF_AVAILABLE = False
RESET_OUTPUT_DIR = False
VERBOSE = False

USE_STRUCTURAL_PROPAGATION_IN_TRAINING = True
USE_STRUCTURAL_PROPAGATION_IN_EVALUATION = True

# Hyperparameters
BATCH_SIZE = 1
EPOCHS = 50
TRAIN_IMAGE_NUM = 400

# Optics configuration
TRAIN_OPTICS_CONFIG = {
    "channel": 1,
    "first_z": 0.005,
    "delta_z": 0.005,
    "layer_num": 1,
    "LCoS_res_h": 1080,
    "LCoS_res_w": 1920,
}

EVAL_OPTICS_CONFIG = {
    "channel": 1,
    "first_z": 0.005,
    "delta_z": 0.005,
    "layer_num": 1,
    "LCoS_res_h": 2160,
    "LCoS_res_w": 3840,
}


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_hyperparams():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if GPU_ID >= device_count:
            raise ValueError(f"GPU_ID={GPU_ID} but only {device_count} CUDA device(s) are available")
        Hyperparams.device = torch.device(f"cuda:{GPU_ID}")
    else:
        Hyperparams.device = torch.device("cpu")

    Hyperparams.SAVE = True
    Hyperparams.batch_size = BATCH_SIZE
    Hyperparams.epochs = EPOCHS
    Hyperparams.train_image_num = TRAIN_IMAGE_NUM
    Hyperparams.learning_rate = 5e-4 * Hyperparams.batch_size * 2
    Hyperparams.total_steps = Hyperparams.epochs * (Hyperparams.train_image_num // Hyperparams.batch_size)
    Hyperparams.learning_rate_step_size = max(1, int(Hyperparams.total_steps * 0.2))
    return Hyperparams.device


def prepare_output_dir(root_path):
    existed = root_path.exists()
    if existed and RESET_OUTPUT_DIR:
        shutil.rmtree(root_path)
        existed = False

    root_path.mkdir(parents=True, exist_ok=True)
    if VERBOSE:
        action = "Using existing" if existed else "Created"
        print(f"{action} output directory: {root_path}")


def build_phase_model(root_path):
    checkpoint = root_path / "models" / "model_state_dict.pt"
    if RESUME_IF_AVAILABLE and checkpoint.exists():
        print(f"Loading checkpoint: {checkpoint}")
        return load_phase_model(
            checkpoint,
            device=Hyperparams.device,
            include_output_activation=False,
        )

    if RESUME_IF_AVAILABLE:
        print(f"Checkpoint not found, starting from a new model: {checkpoint}")
    return TPN_R()


def print_configuration(output_root, eval_path, device):
    print("Dataset-free training")
    print(f"  device: {device}")
    print(f"  output root: {output_root}")
    print(f"  eval path: {eval_path}")
    print(f"  epochs: {Hyperparams.epochs}, batch size: {Hyperparams.batch_size}")
    print(f"  train fields: {Hyperparams.train_image_num}")
    print(
        f"  propagation: train={'structural' if USE_STRUCTURAL_PROPAGATION_IN_TRAINING else 'standard'}, "
        f"eval={'structural' if USE_STRUCTURAL_PROPAGATION_IN_EVALUATION else 'standard'}"
    )


def train_one_config(config_name, data_root, output_root, eval_path, train_optics, eval_optics):
    train_path = data_root / config_name
    root_path = output_root / config_name
    prepare_output_dir(root_path)

    phase_model = build_phase_model(root_path)
    trainer = Trainer(
        train_optics_ins=train_optics,
        eval_optics_ins=eval_optics,
        phs_code_model=phase_model,
        root_path=str(root_path),
        train_path=str(train_path),
        eval_path=str(eval_path),
        PSD_name="/PSD_Tt_1_8_2Value.npy",
        use_structural_propagation_in_training=USE_STRUCTURAL_PROPAGATION_IN_TRAINING,
        use_structural_propagation_in_evaluation=USE_STRUCTURAL_PROPAGATION_IN_EVALUATION,
    )

    print(f"Training: {config_name}")
    trainer.train_dataset_free()
    print(f"Completed: {config_name}")


def main():
    set_seed(SEED)
    device = configure_hyperparams()

    data_root = resolve_repo_path(DATA_ROOT)
    eval_path = resolve_repo_path(EVAL_PATH)
    output_root = resolve_repo_path(OUTPUT_ROOT)

    train_optics = Optics(**TRAIN_OPTICS_CONFIG)
    eval_optics = Optics(**EVAL_OPTICS_CONFIG)

    print_configuration(output_root, eval_path, device)
    for config_name in DATASET_FREE_CONFIGS:
        train_one_config(config_name, data_root, output_root, eval_path, train_optics, eval_optics)


if __name__ == "__main__":
    main()
