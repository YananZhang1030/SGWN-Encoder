# Checkpoints

This directory contains compact pretrained SGWN-Encoder checkpoints exported as plain PyTorch `state_dict` files.

Each checkpoint directory contains:

- `model_state_dict.pt`: tensor-only model weights for `TPN_R`.
- `checkpoint_config.json`: architecture, training source, and checksum metadata.
- `SHA256SUMS`: SHA256 checksum for the state-dict file.

Included checkpoints:

- `sgwn_encoder_dataset_free/`: default dataset-free random-field checkpoint.
- `sgwn_encoder_div2k/`: checkpoint trained on DIV2K training images.
- `sgwn_encoder_mit4k/`: checkpoint trained on MIT4K images.

The original full-object `.pkl` files are intentionally not included.
