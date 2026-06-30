# Script Guide

The top-level scripts are grouped by task and use consistent names.

## Training and Experiments

- `train_dataset_free.py`: train with dataset-free random complex fields.
- `scripts/experiments/train_real_data.py`: train on real RGBD datasets such as MIT4K or DIV2K.
- `scripts/experiments/run_seed_stability.py`: train repeated dataset-free runs with different random seeds.

## Reconstruction

- `scripts/reconstruction/reconstruct_from_hologram_single.py`: reconstruct one existing phase hologram and compare with one RGBD target.
- `scripts/reconstruction/reconstruct_from_hologram_batch.py`: reconstruct a folder of existing phase holograms.

## Evaluation

- `scripts/evaluation/evaluate_dataset_model_distances.py`: compare dataset-trained model folders over a distance list.
- `scripts/evaluation/evaluate_seed_stability_distances.py`: compare seed-stability model folders over a distance list.
- `scripts/evaluation/evaluate_structural_vs_standard_distances.py`: run structural-only vs standard-propagation comparisons.
- `scripts/evaluation/evaluate_checkpoint_key_distances.py`: evaluate one checkpoint at key distances on a RGB-only validation set.
- `scripts/evaluation/evaluate_checkpoint_evaldataset_51dist.py`: thin wrapper around `scripts/evaluation/evaluate_checkpoint_key_distances.py` for the seed-8 51-distance list.

## Analysis and Benchmarking

- `scripts/benchmark/benchmark_sgwn_encoder_2d3d.py`: measure SGWN-Encoder 2D/3D hologram synthesis runtime.
- `scripts/analysis/characterize_psf.py`: characterize structural-only point-spread behavior.

## Core Modules

The following files are imported by the scripts and are not standalone experiment runners:

- `CNNs.py`
- `Optics.py`
- `HolographyDataset.py`
- `Trainer.py`
- `TrainingLogger.py`
- `hyperparams.py`
