"""
Run SGWN-Encoder training stability experiments across fixed random seeds.

This script repeats the dataset-free training with identical architecture,
optical parameters, optimizer settings, batch size, training epochs, and random
field generation strategy. Only the random initialization and DF sampling seed
are varied.
"""

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import csv
import json
import os
import random
import shutil
from datetime import datetime

import numpy as np
import torch

from CNNs import TPN_R
from Optics import Optics
from Trainer import Trainer
from hyperparams import Hyperparams


DEFAULT_SEEDS = [1, 2, 3, 4, 5]


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


def configure_hyperparams(args):
    Hyperparams.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    Hyperparams.SAVE = True
    Hyperparams.ModelPrepared = False
    Hyperparams.batch_size = args.batch_size
    Hyperparams.epochs = args.epochs
    Hyperparams.train_image_num = args.train_image_num
    Hyperparams.learning_rate = args.learning_rate * Hyperparams.batch_size
    Hyperparams.total_steps = Hyperparams.epochs * (Hyperparams.train_image_num // Hyperparams.batch_size)
    Hyperparams.learning_rate_step_size = max(1, int(Hyperparams.total_steps * 0.2))
    Hyperparams.learning_rate_gamma = args.learning_rate_gamma


def read_training_log(run_dir):
    log_path = os.path.join(run_dir, "training_monitor", "training_log.json")
    with open(log_path, "r") as f:
        log = json.load(f)

    epochs = log.get("epochs", [])
    final_epoch = epochs[-1] if epochs else {}
    return {
        "best_epoch": log.get("best_epoch", -1),
        "best_psnr": log.get("best_psnr", 0.0),
        "final_psnr": final_epoch.get("psnr", 0.0),
        "final_ssim": final_epoch.get("ssim", 0.0),
        "final_loss": final_epoch.get("train_loss", 0.0),
        "total_training_time": sum(e.get("training_time", 0.0) for e in epochs),
    }


def write_summary(summary_dir, rows, config):
    os.makedirs(summary_dir, exist_ok=True)

    csv_path = os.path.join(summary_dir, "seed_stability_summary.csv")
    fieldnames = [
        "seed",
        "run_dir",
        "best_epoch",
        "best_psnr",
        "final_psnr",
        "final_ssim",
        "final_loss",
        "total_training_time",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "best_psnr_mean": float(np.mean([r["best_psnr"] for r in rows])),
        "best_psnr_std": float(np.std([r["best_psnr"] for r in rows], ddof=1)),
        "final_psnr_mean": float(np.mean([r["final_psnr"] for r in rows])),
        "final_psnr_std": float(np.std([r["final_psnr"] for r in rows], ddof=1)),
        "final_ssim_mean": float(np.mean([r["final_ssim"] for r in rows])),
        "final_ssim_std": float(np.std([r["final_ssim"] for r in rows], ddof=1)),
        "final_loss_mean": float(np.mean([r["final_loss"] for r in rows])),
        "final_loss_std": float(np.std([r["final_loss"] for r in rows], ddof=1)),
    }
    json_path = os.path.join(summary_dir, "seed_stability_summary.json")
    with open(json_path, "w") as f:
        json.dump({"config": config, "runs": rows, "aggregate": metrics}, f, indent=2)

    md_path = os.path.join(summary_dir, "Supplementary_Note_S10_seed_stability.md")
    with open(md_path, "w") as f:
        f.write("# Supplementary Note S10. Training stability across random seeds\n\n")
        f.write(
            "We repeated SGWN-Encoder training with five random seeds "
            "(1, 2, 3, 4, and 5). All network architecture, optical parameters, "
            "optimizer settings, batch size, training epochs, and dataset-free random-field "
            "generation strategy were kept unchanged. Only the random initialization "
            "and random-field sampling seed were varied.\n\n"
        )
        f.write("| Seed | Best epoch | Best PSNR (dB) | Final PSNR (dB) | Final SSIM | Final loss |\n")
        f.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        for row in rows:
            f.write(
                f"| {row['seed']} | {row['best_epoch']} | {row['best_psnr']:.4f} | "
                f"{row['final_psnr']:.4f} | {row['final_ssim']:.4f} | {row['final_loss']:.4f} |\n"
            )
        f.write("\n")
        f.write(
            f"Across five seeds, the best PSNR was {metrics['best_psnr_mean']:.4f} "
            f"+/- {metrics['best_psnr_std']:.4f} dB, and the final PSNR was "
            f"{metrics['final_psnr_mean']:.4f} +/- {metrics['final_psnr_std']:.4f} dB.\n"
        )


def run_seed(args, seed):
    set_seed(seed)

    experiment_name = f"{args.experiment_name}_seed{seed}"
    root_path = os.path.join(args.output_dir, experiment_name)
    if os.path.exists(root_path):
        if not args.overwrite:
            raise FileExistsError(f"{root_path} already exists. Use --overwrite to replace it.")
        shutil.rmtree(root_path)
    os.makedirs(root_path, exist_ok=True)

    train_optics_ins = Optics(
        args.channel,
        args.first_z,
        args.delta_z,
        args.layer_num,
        LCoS_res_h=args.train_height,
        LCoS_res_w=args.train_width,
    )
    eval_optics_ins = Optics(
        args.channel,
        args.first_z,
        args.delta_z,
        args.layer_num,
        LCoS_res_h=args.eval_height,
        LCoS_res_w=args.eval_width,
    )

    trainer = Trainer(
        train_optics_ins=train_optics_ins,
        eval_optics_ins=eval_optics_ins,
        phs_code_model=TPN_R(),
        root_path=root_path,
        train_path=args.train_path,
        eval_path=args.eval_path,
        PSD_name="/PSD_Tt_1_8_2Value.npy",
        use_structural_propagation_in_training=args.structural_training,
        use_structural_propagation_in_evaluation=args.structural_evaluation,
    )

    manifest = {
        "seed": seed,
        "created_at": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "train_path": args.train_path,
        "eval_path": args.eval_path,
    }
    with open(os.path.join(root_path, "seed_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Starting seed {seed}: {experiment_name}")
    trainer.train_dataset_free()

    row = read_training_log(root_path)
    row.update({"seed": seed, "run_dir": root_path})
    return row


def parse_args():
    parser = argparse.ArgumentParser(description="SGWN-Encoder random-seed stability experiment")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--experiment-name", default="DF_D_R_90_90_L_350_350")
    parser.add_argument("--train-path", default="data/DF_D_R_90_90_L_350_350")
    parser.add_argument("--eval-path", default="./eval")
    parser.add_argument("--output-dir", default="./outputs/seed_stability")
    parser.add_argument("--summary-dir", default="./outputs/seed_stability")
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-image-num", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--learning-rate-gamma", type=float, default=0.5)
    parser.add_argument("--channel", type=int, default=1)
    parser.add_argument("--first-z", type=float, default=0.005)
    parser.add_argument("--delta-z", type=float, default=0.005)
    parser.add_argument("--layer-num", type=int, default=1)
    parser.add_argument("--train-height", type=int, default=1080)
    parser.add_argument("--train-width", type=int, default=1920)
    parser.add_argument("--eval-height", type=int, default=2160)
    parser.add_argument("--eval-width", type=int, default=3840)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--structural-training", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--structural-evaluation", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_hyperparams(args)

    rows = []
    for seed in args.seeds:
        rows.append(run_seed(args, seed))

    config = vars(args)
    config["device_resolved"] = str(Hyperparams.device)
    write_summary(args.summary_dir, rows, config)
    print(f"Seed stability summary saved to {args.summary_dir}")


if __name__ == "__main__":
    main()
