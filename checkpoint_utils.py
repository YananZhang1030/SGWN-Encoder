"""Checkpoint loading helpers for SGWN-Encoder state-dict checkpoints."""

from pathlib import Path
import json

import torch
from torch import nn

from CNNs import TPN_R


def _torch_load_weights(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_checkpoint_path(path):
    path = Path(path)
    if path.is_dir():
        config_path = path / 'checkpoint_config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())
            return path / config.get('state_dict_file', 'model_state_dict.pt')
        return path / 'model_state_dict.pt'
    return path


def load_phase_model(checkpoint_path, device=None, include_output_activation=True):
    """Load a TPN_R checkpoint exported as a plain state_dict.

    Args:
        checkpoint_path: Path to `model_state_dict.pt`, `checkpoint_config.json`, or a model directory.
        device: Torch device or string. Defaults to CPU.
        include_output_activation: Wrap TPN_R with Hardtanh(-pi, pi), matching training.

    Returns:
        A torch.nn.Module ready for evaluation.
    """
    device = torch.device(device or 'cpu')
    checkpoint_path = Path(checkpoint_path)

    config = {}
    if checkpoint_path.name == 'checkpoint_config.json':
        config = json.loads(checkpoint_path.read_text())
        checkpoint_path = checkpoint_path.parent / config.get('state_dict_file', 'model_state_dict.pt')
    else:
        checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        config_path = checkpoint_path.parent / 'checkpoint_config.json'
        if config_path.exists():
            config = json.loads(config_path.read_text())

    state_dict = _torch_load_weights(checkpoint_path, device)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    cleaned_state = {}
    for key, value in state_dict.items():
        if key.startswith('0.'):
            key = key[2:]
        cleaned_state[key] = value

    core = TPN_R().to(device)
    core.load_state_dict(cleaned_state)

    output_activation = config.get('output_activation', 'Hardtanh(-pi, pi)')
    if include_output_activation and output_activation != 'none':
        model = nn.Sequential(core, nn.Hardtanh(-torch.pi, torch.pi)).to(device)
    else:
        model = core
    model.eval()
    return model
