"""
hyperparams.py - Lightweight hyperparameter import module

This file provides default configuration. The values of the Hyperparams class 
can be overridden at runtime by scripts.
Note: Main configuration is now managed in train_dataset_free.py.
"""

import torch

class Hyperparams:
    # Default configuration - can be overridden at runtime by scripts
    batch_size = 1
    TEST_batch = 50
    SAVE = True
    MUL_SAVE = False
    TIME = False
    DIFF = False
    # Depth convention: HolographyDataset training does NOT invert depth maps, so
    # prediction and evaluation must use the same convention. Only set True for
    # checkpoints that were trained with inverted (white=far) depth maps.
    DEPTH_INVERSION = False
    GAMMA = [1, 1, 1]
    time1 = []
    time2 = []
    PSNRSSIM = False
    PSNR = []
    SSIM = []
    init_phs = 0/4
    ModelPrepared = False
    learning_rate = 5e-4 * batch_size * 2
    epochs = 50
    train_image_num = 400
    optimizer = 'Adam'
    total_steps = epochs * (train_image_num // batch_size)
    learning_rate_step_size = max(1, int(total_steps * 0.2))
    learning_rate_gamma = 0.5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # DataLoader configuration
    num_workers = 12
    pin_memory = True
    prefetch_factor = 6