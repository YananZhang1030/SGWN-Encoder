# Structurally Guided Wavefront Neural Encoder for Broad-Range Generalizable 3D Holography

This repository contains the PyTorch implementation of **Structurally Guided Wavefront Neural Encoder for Broad-Range Generalizable 3D Holography**.

SGWN-Encoder is a physics-informed neural hologram encoder for phase-only computer-generated holography (CGH). The implementation follows the revised manuscript terminology: structural propagation is formulated under a scalar Fourier-optics/angular-spectrum-method (ASM) model by analytically removing the spatial-frequency-independent piston phase from the network input while preserving the structural diffraction term. The default model was trained with dataset-free random complex fields and is evaluated across visible wavelengths and propagation distances within the sampled scalar-diffraction regime.

The code supports:

- dataset-free random-field training;
- RGB-D dataset training for comparison experiments;
- phase-only hologram generation and numerical reconstruction;
- structural-propagation and standard-ASM comparison;
- PSF characterization, seed-stability evaluation, and 2D/3D runtime benchmarking.

## Repository Status

This GitHub-ready copy excludes local datasets, raw generated experiment outputs, and full-object training checkpoints. Compact pretrained weights exported as plain PyTorch `state_dict` files are included under `checkpoints/` for reviewer assessment and reproducible examples.

This repository is released under the MIT License. Confirm that the included checkpoint redistribution policy is approved by all relevant authors/institutions before public release.

## Installation

```bash
conda create -n sgwn-encoder-cgh python=3.10
conda activate sgwn-encoder-cgh
pip install -r requirements.txt
```

The original experiments used CUDA-enabled PyTorch. If GPU acceleration is needed, install the PyTorch build matching your CUDA version before or after installing the remaining requirements.

## Project Layout

```text
CNNs.py / Optics.py / HolographyDataset.py
  Core network, scalar ASM optics, and dataset modules.

Trainer.py / TrainingLogger.py / checkpoint_utils.py / hyperparams.py
  Training, evaluation, logging, checkpoint loading, and default runtime settings.

train_dataset_free.py
  Dataset-free random-complex-field training entry point.

predict.py
  Phase-only hologram generation and numerical reconstruction entry point.

scripts/experiments/train_real_data.py
  RGB-D dataset training for dataset-comparison experiments.

scripts/experiments/run_seed_stability.py
  Independent random-seed training runs.

scripts/evaluation/
  Distance generalization, checkpoint evaluation, seed-stability evaluation, and
  structural-vs-standard propagation comparison scripts.

scripts/analysis/characterize_psf.py
  Lateral and axial point-spread-function characterization.

scripts/benchmark/benchmark_sgwn_encoder_2d3d.py
  Native-4K 2D/3D hologram synthesis runtime benchmark.

scripts/reconstruction/
  Numerical reconstruction from already generated phase holograms.
```

Additional script notes are summarized in `SCRIPTS.md`.

## Included Checkpoints

The repository includes compact pretrained checkpoints:

```text
checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt
checkpoints/sgwn_encoder_div2k/model_state_dict.pt
checkpoints/sgwn_encoder_mit4k/model_state_dict.pt
```

These are plain PyTorch `state_dict` files for `TPN_R`, not pickled full model objects. Each checkpoint directory also includes `checkpoint_config.json` and `SHA256SUMS`.

The default checkpoint is:

```text
checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt
```

This model corresponds to the dataset-free random-field training setting described in the manuscript. The DIV2K and MIT4K checkpoints are included for dataset-training comparison experiments.

## Data Layout

External RGB-D datasets should be placed under `data/` using paired image and depth files:

```text
data/<dataset_name>/
  image_0001.png
  image_0001_depth.png
  image_0002.png
  image_0002_depth.png
```

If a depth image is missing, supported evaluation scripts can use a flat zero-depth map.

For a minimal prediction example, place an RGB image and depth map at:

```text
data/example_input/image.png
data/example_input/image_depth.png
```

Then run:

```bash
python predict.py
```

Generated phase holograms, ASM input fields, and numerical reconstructions are written under `pred/`.

## Dataset-Free Training

Dataset-free training is launched from:

```bash
python train_dataset_free.py
```

The default setting uses the random-complex-field configuration:

```text
DF_D_R_90_90_L_350_350
```

Training uses one fixed reference optical configuration and online generated random complex fields. New runs are written under `outputs/train_dataset_free/` so included checkpoints under `checkpoints/` are not overwritten.

## RGB-D Dataset Training

For RGB-D dataset training, set dataset and output paths through environment variables:

```bash
SGWN_ENCODER_DATA_ROOT=./data \
SGWN_ENCODER_MODEL_ROOT=./outputs/train_real_data \
SGWN_ENCODER_EVAL_PATH=./eval \
python scripts/experiments/train_real_data.py
```

The script expects dataset folders such as:

```text
data/MIT4K_500
data/DIV2K_train_HR
```

## Prediction and Reconstruction

`predict.py` uses the dataset-free checkpoint by default and runs structural propagation:

```text
MODEL_PATH = checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt
USE_STRUCTURAL_PROPAGATION = True
```

The propagation mode can be changed in the script. Structural propagation removes the spatial-frequency-independent piston phase from the network input. Standard ASM propagation is available by setting the relevant structural propagation flag to `False`.

The scripts in `scripts/reconstruction/` reconstruct existing 8-bit phase holograms without running the neural encoder.

## Evaluation Scripts

Common evaluation entry points include:

```bash
python scripts/evaluation/evaluate_checkpoint_key_distances.py \
  --model-path checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt \
  --input-dir data/evaluation_images \
  --output-dir outputs/eval_key_distances

python scripts/analysis/characterize_psf.py \
  --model-path checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt \
  --output-dir outputs/psf_characterization/structural_only \
  --device cuda:0

python scripts/benchmark/benchmark_sgwn_encoder_2d3d.py \
  --model-path checkpoints/sgwn_encoder_dataset_free/model_state_dict.pt \
  --output-dir outputs/runtime_benchmark \
  --device cuda:0
```

Use `--help` on individual scripts for all options.

## Data and Outputs

Large local artifacts are ignored by `.gitignore`, including:

- local datasets under `data/` and `datasets/`;
- generated predictions under `pred/`;
- generated experiment outputs under `outputs/`;
- NumPy dumps, logs, and temporary training artifacts.

The public data-release package associated with the manuscript should be used for source data, figure data, and large generated reconstructions.

## Scope and Limitations

This implementation follows the manuscript's scalar Fourier-optics model based on ASM. It does not implement a full-wave electromagnetic solver and does not model evanescent waves, vectorial polarization effects, multiple scattering, material dispersion, or sub-wavelength object-field variations. Reported generalization should therefore be interpreted within the sampled spatial-frequency bandwidth, finite aperture, phase-only encoding capacity, and scalar-diffraction conditions used in the manuscript.

## Citation

The final BibTeX citation will be added after publication. For review-stage use, cite the manuscript title:

```text
Structurally Guided Wavefront Neural Encoder for Broad-Range Generalizable 3D Holography
```
