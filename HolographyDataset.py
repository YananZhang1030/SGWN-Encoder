import os
import random
import re

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from hyperparams import Hyperparams

class HolographyDataset(Dataset):
    """
    Holography dataset for SGWN-Encoder training.
    
    Supports two modes:
    1. Dataset-free decoupled mode (DF_D_...): Randomly generates complex fields
       Pattern: DF_D_[AMP_MODE]_[AMP_SIZE]_[AMP_SCALE]_[PHASE_MODE]_[PHASE_SIZE]_[PHASE_SCALE]
    2. Real dataset: Loads RGB and depth images from folder, supports missing depth maps
    """
    def __init__(self, dataset_path, optics_ins, padding=False):
        """
        Initialize a holography dataset.
        
        Args:
            dataset_path (str): Path to dataset or dataset-free mode string (e.g., 'DF_D_M_50_200_L_50_10')
            optics_ins: Optics instance containing LCoS resolution and other parameters
            padding (bool): Whether to use padding (currently not used)
        """
        self.device = Hyperparams.device
        self.dataset_path = dataset_path
        self.rgbd_res_h = optics_ins.LCoS_res_h
        self.rgbd_res_w = optics_ins.LCoS_res_w
        self.optics_ins = optics_ins
        self.padding = padding
        self.ROTATE = True
      
        # Check dataset type: supports dataset-free decoupled mode or real data
        # Regex pattern: DF_D_[AMP]_[SIZE]_[SCALE]_[PHASE]_[SIZE]_[SCALE]
        if re.search(r'DF_D_\w+_\d+_\d+_\w+_\d+_\d+', self.dataset_path):
            # Dataset-free random generation mode
            self.dataset_files = None
            self.random_data_size = Hyperparams.train_image_num
        else:
            # Real dataset mode
            self.dataset_files = self._load_files(self.dataset_path)

    def _load_files(self, path):
        """
        Load RGB and depth image file pairs from given path.
        
        Args:
            path (str): Directory path containing image files
            
        Returns:
            list or None: List of (rgb_path, depth_path) tuples, or None if path doesn't exist
                         depth_path is None if corresponding depth file doesn't exist
        """
        if not os.path.exists(path):
            return None
      
        # Exclude files ending with '_depth.png', only get regular RGB images as main index
        pattern = re.compile(r'_depth\.png$')
        img_files = [f for f in os.listdir(path) 
                     if not pattern.search(f) and 
                        os.path.isfile(os.path.join(path, f)) and
                        f.endswith('.png')]
      
        valid_files = []
        for f in img_files:
            depth_file_name = f.replace('.png', '_depth.png')
            depth_path = os.path.join(path, depth_file_name)
            full_img_path = os.path.join(path, f)
          
            # Check if corresponding depth file exists
            # If exists, record path; if not, mark as None (will be auto-filled with zeros later)
            if os.path.exists(depth_path):
                valid_files.append((full_img_path, depth_path))
            else:
                valid_files.append((full_img_path, None))
      
        return valid_files if valid_files else None

    # ==========================================
    # Amplitude Generators - Decoupled Mode Only
    # ==========================================
    def _generate_amplitude(self, amp_mode, amp_param, amp_scale):
        """
        Generate amplitude based on specified mode.
        
        Args:
            amp_mode (str): Amplitude generation mode ('multilayer', 'rayleigh', 'uniform')
            amp_param (int): Initial size parameter
            amp_scale (int): Scale parameter
            
        Returns:
            np.ndarray: Generated amplitude image (uint8, shape: [H, W])
        """
        if amp_mode == 'multilayer':
            return self._generate_multilayer_amplitude_decoupled(amp_param, amp_scale)
        elif amp_mode == 'rayleigh':
            return self._generate_rayleigh_amplitude_decoupled(amp_param, amp_scale)
        elif amp_mode == 'uniform':
            return self._generate_uniform_amplitude_decoupled(amp_param, amp_scale)
        else:
            raise ValueError(f"Unknown amplitude mode: {amp_mode}")
  
    def _generate_multilayer_amplitude_decoupled(self, initial_size, scale):
        """
        Generate multilayer amplitude pattern.
        
        Creates 3 layers with different interpolation methods and adds noise.
        """
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height_variations = np.round(np.random.normal(mean_size, std_size, 3)).astype(int).clip(1, self.rgbd_res_h)
        width_variations = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h, 3)).astype(int).clip(1, self.rgbd_res_w)
      
        # Generate 3 layers
        img0 = np.random.randint(0, 256, (height_variations[0], width_variations[0]), dtype=np.uint8)
        img0 = cv2.resize(img0, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
        img1 = np.random.randint(0, 256, (height_variations[1], width_variations[1]), dtype=np.uint8)
        img1 = cv2.resize(img1, (self.rgbd_res_w, self.rgbd_res_h), interpolation=1)
        img2 = np.random.randint(0, 256, (height_variations[2], width_variations[2]), dtype=np.uint8)
        img2 = cv2.resize(img2, (self.rgbd_res_w, self.rgbd_res_h), interpolation=2)
      
        # Add noise
        noise_intensity = scale * 0.1
        img_noise = np.random.randint(0, 256, (random.randint(1, 540), random.randint(1, 960)), dtype=np.uint8)
        img_noise = cv2.resize(img_noise, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
      
        # Composite layers
        modulo_param = max(100, int(200 - scale * 10))
        img = img0 + img1 % random.randint(100, modulo_param) + img2 % random.randint(100, modulo_param) + img_noise * noise_intensity
      
        if self.ROTATE:
            img = (img + random.randint(0, 256)) % 256
          
        return img.astype(np.uint8)
  
    def _generate_rayleigh_amplitude_decoupled(self, initial_size, scale):
        """
        Generate amplitude using Rayleigh distribution.
        
        Args:
            initial_size (int): Initial size for random generation
            scale (int): Scale parameter for Rayleigh distribution
            
        Returns:
            np.ndarray: Generated amplitude image (uint8)
        """
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height = np.round(np.random.normal(mean_size, std_size)).astype(int).clip(1, self.rgbd_res_h)
        width = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h)).astype(int).clip(1, self.rgbd_res_w)
      
        rayleigh_scale = scale / 100.0
        amp = np.random.rayleigh(scale=rayleigh_scale, size=(height, width))
        amp = np.clip(amp * 255 / (3 * rayleigh_scale), 0, 255).astype(np.uint8)

        amp = cv2.resize(amp, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
        return amp
  
    def _generate_uniform_amplitude_decoupled(self, initial_size, scale):
        """
        Generate amplitude using uniform distribution.
        
        Args:
            initial_size (int): Initial size for random generation
            scale (int): Scale parameter controlling maximum value (0-100)
            
        Returns:
            np.ndarray: Generated amplitude image (uint8)
        """
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height = np.round(np.random.normal(mean_size, std_size)).astype(int).clip(1, self.rgbd_res_h)
        width = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h)).astype(int).clip(1, self.rgbd_res_w)
      
        max_val = min(255, int(scale * 255 / 100))
        img = np.random.randint(0, max_val + 1, (height, width), dtype=np.uint8)
        img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
        if self.ROTATE:
            img = (img + random.randint(0, 256)) % 256
        return img.astype(np.uint8)

    # ==========================================
    # Phase Generators - Decoupled Mode Only
    # ==========================================
    def _generate_phase(self, phase_mode, phase_param, phase_scale):
        """
        Generate phase based on specified mode.
        
        Args:
            phase_mode (str): Phase generation mode ('laplace', 'gaussian', 'mixed')
            phase_param (int): Initial size parameter
            phase_scale (int): Scale parameter
            
        Returns:
            np.ndarray: Generated phase image (uint8, shape: [H, W])
        """
        if phase_mode == 'laplace':
            return self._generate_laplace_phase_decoupled(phase_param, phase_scale)
        elif phase_mode == 'gaussian':
            return self._generate_gaussian_phase_decoupled(phase_param, phase_scale)
        elif phase_mode == 'mixed':
            return self._generate_mixed_phase_decoupled(phase_param, phase_scale)
        else:
            raise ValueError(f"Unknown phase mode: {phase_mode}")
  
    def _generate_laplace_phase_decoupled(self, initial_size, scale):
        """
        Generate phase using Laplace distribution.
        
        Args:
            initial_size (int): Initial size for random generation
            scale (int): Scale parameter for Laplace distribution
            
        Returns:
            np.ndarray: Generated phase image (uint8)
        """
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height = np.round(np.random.normal(mean_size, std_size)).astype(int).clip(1, self.rgbd_res_h)
        width = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h)).astype(int).clip(1, self.rgbd_res_w)
      
        laplace_scale = scale / 10.0
        laplace_noise = np.random.laplace(loc=127.5, scale=laplace_scale, size=(height, width))
        clipped_phase = cv2.resize(laplace_noise, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
        clipped_phase = np.clip(clipped_phase, 0, 255)
        return clipped_phase.astype(np.uint8)
  
    def _generate_gaussian_phase_decoupled(self, initial_size, scale):
        """
        Generate phase using Gaussian distribution.
        
        Args:
            initial_size (int): Initial size for random generation
            scale (int): Standard deviation for Gaussian distribution
            
        Returns:
            np.ndarray: Generated phase image (uint8)
        """
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height = np.round(np.random.normal(mean_size, std_size)).astype(int).clip(1, self.rgbd_res_h)
        width = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h)).astype(int).clip(1, self.rgbd_res_w)
      
        gaussian_noise = np.random.normal(loc=127.5, scale=scale, size=(height, width))
        clipped_image = np.clip(gaussian_noise, 0, 255)
        clipped_image = cv2.resize(clipped_image, (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
        return clipped_image.astype(np.uint8)
  
    def _generate_mixed_phase_decoupled(self, initial_size, scale):
        """
        Generate phase using mixed distribution (Laplace + Gaussian + Triangular).
        
        Creates 3 layers with different distributions and averages them.
        
        Args:
            initial_size (int): Initial size for random generation
            scale (int): Standard deviation parameter
            
        Returns:
            np.ndarray: Generated phase image (uint8)
        """
        target_mean = 127.5
        target_std = scale
      
        mean_size = initial_size
        std_size = int(mean_size / 2)
        height_variations = np.round(np.random.normal(mean_size, std_size, 3)).astype(int).clip(1, self.rgbd_res_h)
        width_variations = np.round(np.random.normal(mean_size * self.rgbd_res_w / self.rgbd_res_h, std_size * self.rgbd_res_w / self.rgbd_res_h, 3)).astype(int).clip(1, self.rgbd_res_w)
      
        # Layer 0: Laplace distribution
        laplace_scale = target_std / 1.414
        layer0_raw = np.random.laplace(loc=target_mean, scale=laplace_scale, size=(height_variations[0], width_variations[0]))
        layer0 = cv2.resize(np.clip(layer0_raw, 0, 255).astype(np.uint8), (self.rgbd_res_w, self.rgbd_res_h), interpolation=0)
      
        # Layer 1: Gaussian distribution
        layer1_raw = np.random.normal(loc=target_mean, scale=target_std, size=(height_variations[1], width_variations[1]))
        layer1 = cv2.resize(np.clip(layer1_raw, 0, 255).astype(np.uint8), (self.rgbd_res_w, self.rgbd_res_h), interpolation=1)
      
        # Layer 2: Triangular distribution
        triangle_width = target_std * 2.0
        left = max(0, target_mean - triangle_width)
        right = min(255, target_mean + triangle_width)
        layer2_raw = np.random.triangular(left=left, mode=target_mean, right=right, size=(height_variations[2], width_variations[2]))
        layer2 = cv2.resize(layer2_raw.astype(np.uint8), (self.rgbd_res_w, self.rgbd_res_h), interpolation=2)
      
        # Average the three layers
        result = (layer0.astype(np.float32) + layer1.astype(np.float32) + layer2.astype(np.float32)) / 3.0
        result_final = np.clip(result, 0, 255).astype(np.uint8)
      
        return result_final

    # ==========================================
    # Core Logic and Dataset-Free Entry Point
    # ==========================================
    def _parse_dataset_free_mode(self):
        """
        Parse dataset-free decoupled generation mode from dataset path.
        
        Pattern: DF_D_[AMP_MODE]_[AMP_SIZE]_[AMP_SCALE]_[PHASE_MODE]_[PHASE_SIZE]_[PHASE_SCALE]
        Mode codes: R=rayleigh, L=laplace, M=multilayer, G=gaussian, X=mixed, U=uniform
        
        Returns:
            tuple: (amp_mode, phase_mode, amp_param, amp_scale, phase_param, phase_scale)
                   Default values if pattern doesn't match
        """
        path = self.dataset_path
        decoupled_pattern = r'DF_D_([RLMGXU])_(\d+)_(\d+)_([RLMGXU])_(\d+)_(\d+)$'
        match = re.search(decoupled_pattern, path)
      
        if match:
            mode_mapping = {
                'R': 'rayleigh', 'L': 'laplace', 'M': 'multilayer',
                'G': 'gaussian', 'X': 'mixed', 'U': 'uniform'
            }
            return (
                mode_mapping.get(match.group(1), 'multilayer'), 
                mode_mapping.get(match.group(4), 'laplace'),  
                int(match.group(2)), int(match.group(3)), 
                int(match.group(5)), int(match.group(6))
            )
        # Default values
        return 'multilayer', 'laplace', 50, 200, 50, 10

    def _generate_dataset_free_field(self):
        """
        Dataset-free generation entry point - only supports decoupled logic.
        
        Generates complex field by combining amplitude and phase.
        
        Returns:
            tuple: (real, imag) - Real and imaginary parts of complex field
        """
        amp_mode, phase_mode, amp_param, amp_scale, phase_param, phase_scale = self._parse_dataset_free_mode()
      
        amp = self._generate_amplitude(amp_mode, amp_param, amp_scale)
        phase = self._generate_phase(phase_mode, phase_param, phase_scale)
      
        # Normalize and compose complex field
        amp = amp.astype(np.float32) / 255.0
        phase = (phase.astype(np.float32) / 255.0) * 2 * np.pi - np.pi
      
        real = amp * np.cos(phase)
        imag = amp * np.sin(phase)
      
        return real, imag

    def __len__(self):
        """
        Return dataset size.
        
        Returns:
            int: Dataset size. For dataset-free mode, returns random_data_size; 
                 for real dataset, returns number of files.
        """
        if self.dataset_files is None:
            # If in random generation mode, return preset size
            if hasattr(self, 'random_data_size'):
                return self.random_data_size
            return 0
        return len(self.dataset_files)

    def __getitem__(self, idx):
        """
        Get a single training sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            For dataset-free mode: (real_slm, imag_slm, rgb_path)
            For real dataset: (img, image_ints_ts, image_amp_slm, image_depth_slm, rgb_path)
        """
        depth_num = self.optics_ins.layer_num

        # 1. Dataset-free decoupled format - randomly generate complex field
        if re.search(r'DF_D_\w+_\d+_\d+_\w+_\d+_\d+', self.dataset_path):
            real, imag = self._generate_dataset_free_field()
            rgb_path = 'dataset_free_random_field'
            real_slm = torch.Tensor(real).unsqueeze(0)
            imag_slm = torch.Tensor(imag).unsqueeze(0)
            return real_slm, imag_slm, rgb_path

        # 2. Real dataset mode
        elif self.dataset_files and len(self.dataset_files) > 0:
            rgb_path, depth_path = self.dataset_files[idx]
          
            # Read RGB image
            img = cv2.imread(rgb_path)[:, :, self.optics_ins.channel]
          
            # Read or generate depth map
            if depth_path is not None:
                image_depth = cv2.imread(depth_path, 0)
            else:
                # Missing depth map, automatically generate all-zero image (black image)
                image_depth = np.zeros_like(img, dtype=np.uint8)
          
        else:
            raise ValueError(f"Unknown dataset format: {self.dataset_path}")

        # Center crop to target resolution. RGB and depth use the same crop window.
        img_h, img_w = img.shape[:2]
        if img_h >= self.rgbd_res_h and img_w >= self.rgbd_res_w:
            top = (img_h - self.rgbd_res_h) // 2
            left = (img_w - self.rgbd_res_w) // 2
            img = img[top:top + self.rgbd_res_h, left:left + self.rgbd_res_w]
            image_depth = image_depth[top:top + self.rgbd_res_h, left:left + self.rgbd_res_w]
        elif img_h != self.rgbd_res_h or img_w != self.rgbd_res_w:
            img = cv2.resize(img, (self.rgbd_res_w, self.rgbd_res_h))
            image_depth = cv2.resize(image_depth, (self.rgbd_res_w, self.rgbd_res_h))

        # Convert to Tensor and physical quantities
        image = img.astype(np.float32) / 255
        image_ints_ts = torch.Tensor(image).unsqueeze(0)
      
        # Amplitude is square root of intensity
        image_amp = np.sqrt(image)
        image_amp_slm = torch.Tensor(image_amp).unsqueeze(0)
      
        # Quantize depth map (simulate multi-layer SLM/LCoS)
        image_depth = image_depth.astype(np.float32) / 255
        for dep in range(depth_num):
            image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
        image_depth_slm = torch.Tensor(image_depth).unsqueeze(0)
      
        return img, image_ints_ts, image_amp_slm, image_depth_slm, rgb_path
