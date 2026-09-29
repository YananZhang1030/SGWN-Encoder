import json
import os
import shutil
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from HolographyDataset import HolographyDataset
from TrainingLogger import TrainingLogger
from checkpoint_utils import load_phase_model
from hyperparams import Hyperparams



class Trainer:
    """
    Trainer class for CGH (Computer-Generated Hologram) training and evaluation.
    
    Handles training of phase encoding networks, evaluation on datasets, and prediction.
    Supports both regular RGBD dataset training and dataset-free random-field training.
    """
    def __init__(self, train_optics_ins, eval_optics_ins, phs_code_model=None, root_path='', train_path='', eval_path='', PSD_name='/PSD_EVAL.npy', use_structural_propagation_in_training=True, use_structural_propagation_in_evaluation=False):
        """
        Initialize Trainer.
        
        Args:
            train_optics_ins: Optics instance for training
            eval_optics_ins: Optics instance for evaluation
            phs_code_model: Phase encoding model (None for prediction-only mode)
            root_path: Root directory for saving models and logs
            train_path: Path to training dataset
            eval_path: Path to evaluation dataset
            PSD_name: Name of PSD data file
            use_structural_propagation_in_training: Whether to use structural propagation in training
            use_structural_propagation_in_evaluation: Whether to use structural propagation in evaluation
        """
        self.train_path = train_path
        self.eval_path = eval_path
        self.root_path = root_path
        self.use_structural_propagation_in_training = use_structural_propagation_in_training
        self.use_structural_propagation_in_evaluation = use_structural_propagation_in_evaluation
        if root_path:
            self.model_best_path = os.path.join(self.root_path, 'model_state_dict.pt')
            self.PSD_path = self.root_path + PSD_name
        else:
            self.model_best_path = ''
            self.PSD_path = ''

        self.train_optics_ins = train_optics_ins
        self.eval_optics_ins = eval_optics_ins
        # Use eval_optics_ins if train_optics_ins is None (prediction-only mode)
        optics_for_resolution = train_optics_ins if train_optics_ins is not None else eval_optics_ins
        self.rgbd_res_h = optics_for_resolution.LCoS_res_h
        self.rgbd_res_w = optics_for_resolution.LCoS_res_w
        self.device = Hyperparams.device

        if train_path:
            self.train_Dataset_ins = HolographyDataset(self.train_path, self.train_optics_ins)
        if eval_path:
            self.eval_Dataset_ins = HolographyDataset(self.eval_path, self.eval_optics_ins)

        self.best_loss = 99999
        self.psnr_best = 0
        self.phs_net = None

        # Initialize network and optimizer only if model is provided (i.e., training mode)
        if phs_code_model:
            self.phs_net = nn.Sequential(
                phs_code_model.to(self.device),
                nn.Hardtanh(-torch.pi, torch.pi)
            )
            self.loss = nn.L1Loss().to(self.device)
            self.optimizer = torch.optim.Adam(self.phs_net.parameters(), lr=Hyperparams.learning_rate)
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, Hyperparams.learning_rate_step_size, Hyperparams.learning_rate_gamma)
            
            # Initialize training monitor
            if self.root_path:
                experiment_name = os.path.basename(self.root_path)
                self.logger = TrainingLogger(self.root_path, experiment_name)
                
                # Save configuration snapshot
                self.logger.save_config_snapshot(train_optics_ins, train_path, eval_path, self.phs_net)

    @staticmethod
    def _align_reconstruction_tensors(target_amp, recon_amp, use_ols=True, eps=1e-10):
        """
        Align target and reconstruction tensors before metric computation.

        Args:
            target_amp: Target amplitude tensor [B, C, H, W].
            recon_amp: Reconstructed amplitude tensor [B, C, H, W].
            use_ols: If True, fit the reconstruction scale with least squares.
            eps: Numerical stability term.

        Returns:
            target_norm: Target normalized by its own maximum.
            recon_aligned: Scaled and clamped reconstruction.
            scale_s: Fitted scale coefficient [B, 1, 1, 1].
        """
        reduce_dims = tuple(range(1, target_amp.dim()))
        target_max = torch.amax(target_amp, dim=reduce_dims, keepdim=True)
        target_norm = target_amp / (target_max + eps)

        if use_ols:
            numerator = torch.sum(target_norm * recon_amp, dim=reduce_dims, keepdim=True)
            denominator = torch.sum(recon_amp.square(), dim=reduce_dims, keepdim=True)
            scale_s = numerator / (denominator + eps)
            recon_aligned = torch.clamp(recon_amp * scale_s, 0.0, 1.0)
            return target_norm, recon_aligned, scale_s

        recon_max = torch.amax(recon_amp, dim=reduce_dims, keepdim=True)
        scale_s = torch.ones_like(target_max)
        recon_aligned = torch.clamp(recon_amp / (recon_max + eps), 0.0, 1.0)
        return target_norm, recon_aligned, scale_s

    def train(self):
        """
        Train the phase encoding network on RGBD dataset.

        Performs training loop with backward propagation, forward reconstruction, and evaluation.
        """
        if len(self.train_Dataset_ins) == 0:
            raise FileNotFoundError(f"No training images found in: {self.train_path}")
        if self.PSD_path:
            self.delete_train_data(self.PSD_path)  # Start metric accumulation from a clean file
        for epoch in tqdm(range(Hyperparams.epochs), desc="Epochs"):
            # Training mode
            self.phs_net.train()
            loss_epoch = 0
            coefficient = 0
            # Record start time
            start_time = time.time()
            train_loader = DataLoader(
                    self.train_Dataset_ins, 
                    batch_size=Hyperparams.batch_size, 
                    shuffle=True,
                    num_workers=8,
                    pin_memory=True,
                    prefetch_factor=4  # Number of batches to prefetch per worker
                   )
            # Iterate batches using DataLoader
            for batch_idx, (_, _, image_amp_slm, image_depth_slm, _) in tqdm(enumerate(train_loader),
                                                                                         total=len(train_loader),
                                                                                         desc="Training Batches",
                                                                                         position=1,
                                                                                         leave=True,
                                                                                         dynamic_ncols=True,
                                                                                         miniters=5):
                image_amp_slm = image_amp_slm.to(self.device)
                image_depth_slm = image_depth_slm.to(self.device)
                u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
                _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
                depth_num = self.train_optics_ins.layer_num

                # Select global phase kernel for backward propagation based on flag
                if self.use_structural_propagation_in_training:
                    h_backward_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                    h_backward_delta_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                else:
                    h_backward_g_selected = self.train_optics_ins.h_backward_g
                    h_backward_delta_g_selected = self.train_optics_ins.h_backward_delta_g

                for dep in range(depth_num):
                    if dep != 0:
                        u[image_depth_slm == (depth_num - dep - 1) / depth_num] = _u[
                            image_depth_slm == (depth_num - dep - 1) / depth_num]
                    if dep == depth_num - 1:
                        u = self.train_optics_ins.prop_asm(self.train_optics_ins.h_backward_s, h_backward_g_selected, u0=u)
                    else:
                        u = self.train_optics_ins.prop_asm(self.train_optics_ins.h_backward_delta_s, h_backward_delta_g_selected, u0=u)

                # Complex amplitude
                u = u / torch.max(torch.abs(u))
                
                depth_num = self.train_optics_ins.layer_num
                self.optimizer.zero_grad()  # Reset gradients to zero
                phs_slm = self.phs_net(u)
                phs_slm = phs_slm - self.train_optics_ins.phase_grating

                # Holographic reconstruction
                rec_amp_slm = torch.zeros_like(image_amp_slm)
                rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))

                # Select global phase component based on flag
                if self.use_structural_propagation_in_training:
                    h_g = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                    h_delta_g = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                else:
                    h_g = self.train_optics_ins.h_forward_g
                    h_delta_g = self.train_optics_ins.h_forward_delta_g

                for dep in range(depth_num):
                    if dep == 0:
                        rec_u = self.train_optics_ins.prop_asm(self.train_optics_ins.h_forward_s, h_g, phs_in=phs_slm)
                    else:
                        rec_u = self.train_optics_ins.prop_asm(self.train_optics_ins.h_forward_delta_s, h_delta_g, u0=rec_u)
                    rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                        image_depth_slm == (depth_num - dep - 1) / depth_num]

                coefficient_temp = torch.sum(image_amp_slm) / torch.sum(rec_amp_slm)
                coefficient_temp = torch.clamp(coefficient_temp, 0.7, 1.5)
                coefficient = coefficient + coefficient_temp

                loss_val = self.loss(coefficient_temp * rec_amp_slm, image_amp_slm)

                
                loss_val.backward()  # Backpropagate loss

                self.optimizer.step()
                self.scheduler.step()
                loss_epoch = loss_epoch + loss_val.item()

            coefficient_avg = coefficient / len(self.train_Dataset_ins)
            end_time = time.time()
            training_time = end_time - start_time
            print("Train {} | Epoch {}/{} : Training Loss: {:.4f}, Coefficient_Avg:{:.4f}, Time: {:.2f} seconds".format(
                    os.path.splitext(os.path.basename(self.root_path))[0],
                    epoch, Hyperparams.epochs, loss_epoch, coefficient_avg.item(), training_time))

            psnr_avg, ssim_avg, eval_results = self.evaluate(epoch)

            # Log training data; without evaluation data, best is selected by training loss
            is_best = self.logger.log_epoch(epoch, loss_epoch, coefficient_avg.item(), psnr_avg, ssim_avg, training_time,
                                            eval_available=bool(eval_results))
            
            # Save model
            self.logger.save_model(self.phs_net, epoch, is_best=is_best, is_latest=True)
            
            # If best result, update best value and save to original path
            if is_best:
                self.psnr_best = psnr_avg
                torch.save(self.phs_net.state_dict(), self.model_best_path)  # Save plain state_dict checkpoint
                # Save best results (holograms and reconstructions) to best_results directory, reusing evaluate() results
                if eval_results:
                    self.save_best_epoch_results(eval_results, epoch, psnr_avg, ssim_avg, coefficient_avg.item())
                    
            # Plot training curves
            self.logger.plot_training_curves()

        # Cleanup after training completes
        self.logger.save_training_log()
        print(f"Training completed. Best PSNR: {self.psnr_best:.4f} dB")


    def train_dataset_free(self):
        """
        Train the phase encoding network on dataset-free random complex fields.
        
        Uses generated complex fields directly instead of RGBD images.
        """
        if len(self.train_Dataset_ins) == 0:
            raise FileNotFoundError(f"No training data found in: {self.train_path}")
        if self.PSD_path:
            self.delete_train_data(self.PSD_path)  # Start metric accumulation from a clean file

        # Collect dataset-free samples in the first round
        if hasattr(self, 'train_Dataset_ins'):
            self.collect_and_save_dataset_free_samples()
        
        for epoch in tqdm(range(Hyperparams.epochs), desc="Epochs"):
            # Training mode
            self.phs_net.train()
            loss_epoch = 0
            coefficient = 0
            # Record start time
            start_time = time.time()
            train_loader = DataLoader(
                    self.train_Dataset_ins, 
                    batch_size=Hyperparams.batch_size, 
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False,  # Speeds up data transfer but may increase memory pressure
                    prefetch_factor=2  # Number of batches to prefetch per worker
                   )
            # Iterate batches using DataLoader
            for batch_idx, (real_slm, imag_slm, _) in tqdm(enumerate(train_loader),
                                                    total=len(train_loader),
                                                    desc="Training Batches",
                                                    position=1,
                                                    leave=True,
                                                    dynamic_ncols=True,
                                                    miniters=5):
                

                real_slm = real_slm.to(self.device)
                imag_slm = imag_slm.to(self.device)
                u_data = torch.complex(real_slm, imag_slm) 

                # Complex amplitude
                u = u_data / torch.max(torch.abs(u_data))
                depth_num = self.train_optics_ins.layer_num
                self.optimizer.zero_grad()  # Reset gradients to zero
                phs_slm = self.phs_net(u)
                phs_slm = phs_slm - self.train_optics_ins.phase_grating
                
                # Disable gradient computation for prop_asm
                with torch.no_grad():
                    rec_u_abs = torch.abs(self.train_optics_ins.prop_asm(self.train_optics_ins.h_forward_s, self.train_optics_ins.h_forward_g, u0=u_data))  # No gradient computation

                # Select global phase component based on flag
                if self.use_structural_propagation_in_training:
                    h_g = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                else:
                    h_g = self.train_optics_ins.h_forward_g
                
                rec_holo_abs = torch.abs(self.train_optics_ins.prop_asm(self.train_optics_ins.h_forward_s, h_g, phs_in=phs_slm))

                coefficient_temp = torch.sum(rec_u_abs) / torch.sum(rec_holo_abs)
                coefficient_temp = torch.clamp(coefficient_temp, 0.7, 1.5)
                coefficient = coefficient + coefficient_temp

                # Calculate loss value
                loss_val = self.loss(coefficient_temp * rec_holo_abs, rec_u_abs)
                
                loss_val.backward()  # Backpropagate loss
                self.optimizer.step()
                self.scheduler.step()
                loss_epoch = loss_epoch + loss_val.item()

                if self.device.type == "cuda":
                    torch.cuda.empty_cache()  # Clear GPU cache after each batch to prevent OOM
                    torch.cuda.synchronize()  # Wait for all devices to complete work

                
            coefficient_avg = coefficient / len(self.train_Dataset_ins)
            end_time = time.time()
            training_time = end_time - start_time
            print("Train {} | Epoch {}/{} : Training Loss: {:.4f}, Coefficient_Avg:{:.4f}, Time: {:.2f} seconds".format(
                    os.path.splitext(os.path.basename(self.root_path))[0],
                    epoch, Hyperparams.epochs, loss_epoch, coefficient_avg.item(), training_time))

            psnr_avg, ssim_avg, eval_results = self.evaluate(epoch)        
            
            # Log training data; without evaluation data, best is selected by training loss
            is_best = self.logger.log_epoch(epoch, loss_epoch, coefficient_avg.item(), psnr_avg, ssim_avg, training_time,
                                            eval_available=bool(eval_results))
            
            # Save model
            self.logger.save_model(self.phs_net, epoch, is_best=is_best, is_latest=True)
            
            # If best result, update best value and save to original path
            if is_best:
                self.psnr_best = psnr_avg
                torch.save(self.phs_net.state_dict(), self.model_best_path)  # Save plain state_dict checkpoint
                # Save best results (holograms and reconstructions) to best_results directory, reusing evaluate() results
                if eval_results:
                    self.save_best_epoch_results(eval_results, epoch, psnr_avg, ssim_avg, coefficient_avg.item())
                    
            # Plot training curves
            self.logger.plot_training_curves()
            
        # Cleanup after training completes
        self.logger.save_training_log()
        print(f"Training completed. Best PSNR: {self.psnr_best:.4f} dB")
            
    def collect_and_save_dataset_free_samples(self):
        """
        Collect and save dataset-free training samples.
        
        Extracts real, imaginary, amplitude, and phase components from generated complex fields.
        """
        try:
            train_loader = DataLoader(
                self.train_Dataset_ins,
                batch_size=1,  # Process only 1 sample at a time
                shuffle=False,
                num_workers=1
            )
            
            samples = []
            for i, (real_slm, imag_slm, _) in enumerate(train_loader):
                if i >= 5:  # Take only first 5 samples
                    break
                    
                # Convert to complex
                real_slm = real_slm.to(self.device)
                imag_slm = imag_slm.to(self.device)
                u_data = torch.complex(real_slm, imag_slm)
                u_normalized = u_data / torch.max(torch.abs(u_data))
                
                # Extract real part, imaginary part, amplitude, and phase
                real_part = torch.real(u_normalized).cpu().detach().squeeze().numpy()
                imag_part = torch.imag(u_normalized).cpu().detach().squeeze().numpy()
                amplitude = torch.abs(u_normalized).cpu().detach().squeeze().numpy()
                phase = torch.angle(u_normalized).cpu().detach().squeeze().numpy()
                
                # Convert to 8-bit images
                real_img = ((real_part + 1) * 127.5).clip(0, 255).astype(np.uint8)
                imag_img = ((imag_part + 1) * 127.5).clip(0, 255).astype(np.uint8)
                amp_img = (amplitude * 255).clip(0, 255).astype(np.uint8)
                phase_img = ((phase + np.pi) / (2 * np.pi) * 255).clip(0, 255).astype(np.uint8)
                
                sample = {
                    'real': real_img,
                    'imag': imag_img,
                    'amp': amp_img,
                    'phase': phase_img
                }
                samples.append(sample)
                
            # Save sample information
            sample_info = {
                'total_samples': len(samples),
                'data_type': 'dataset_free_complex',
                'normalization': 'max_absolute_normalization',
                'format': 'real_imag_amplitude_phase',
                'value_range': {
                    'real_imag': '[-1, 1] -> [0, 255]',
                    'amplitude': '[0, 1] -> [0, 255]',
                    'phase': '[-pi, pi] -> [0, 255]'
                }
            }
            
            self.logger.save_dataset_free_samples(samples, sample_info)
            
        except Exception as e:
            print(f"Warning: dataset-free sample collection failed: {e}")
            
    def save_best_epoch_results(self, results, epoch, psnr_avg, ssim_avg, coefficient_avg):
        """
        Save all results of the best epoch to best_results directory, named by original image name.
        
        Args:
            results: List of evaluation results
            epoch: Epoch number
            psnr_avg: Average PSNR
            ssim_avg: Average SSIM
            coefficient_avg: Average coefficient
        """
        try:
            best_results_dir = self.logger.dirs['best_results']
            
            # Save results for each evaluation image
            metrics_summary = []
            for result in results:
                base_name = result['filename']
                
                # Save hologram (phase map)
                hologram_path = os.path.join(best_results_dir, f'{base_name}_hologram.png')
                cv2.imwrite(hologram_path, result['hologram'])
                
                # Save reconstruction image
                reconstruction_path = os.path.join(best_results_dir, f'{base_name}_reconstruction.png')
                cv2.imwrite(reconstruction_path, result['reconstruction'])
                
                # Collect metrics information
                metrics_summary.append({
                    'filename': base_name,
                    'psnr': result['metrics']['psnr'],
                    'ssim': result['metrics']['ssim'],
                    'coefficient': result['metrics']['coefficient']
                })
            
            # Use passed average metrics directly, don't recalculate
            # Save detailed metrics information
            summary_info = {
                'epoch': epoch,
                'total_images': len(results),
                'average_metrics': {
                    'psnr': float(psnr_avg),
                    'ssim': float(ssim_avg),
                    'coefficient': float(coefficient_avg)
                },
                'individual_results': metrics_summary
            }
            
            with open(os.path.join(best_results_dir, 'best_epoch_metrics.json'), 'w') as f:
                json.dump(summary_info, f, indent=2)
            
            # Save human-readable summary file
            with open(os.path.join(best_results_dir, 'best_epoch_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Best Epoch Results Summary - Epoch {epoch}\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Number of evaluation images: {len(results)}\n")
                f.write(f"Average PSNR: {psnr_avg:.4f} dB\n")
                f.write(f"Average SSIM: {ssim_avg:.4f}\n")
                f.write(f"Average energy coefficient: {coefficient_avg:.4f}\n\n")
                f.write("Individual image results:\n")
                for metrics in metrics_summary:
                    f.write(f"  {metrics['filename']}: PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}\n")
            
            print(f"Saved best epoch {epoch} results: {len(results)} images to {best_results_dir}")
            
        except Exception as e:
            print(f"Warning: failed to save best epoch results: {e}")
                    
    
    def evaluate(self, epoch):
        """
        Evaluate the model on evaluation dataset.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            tuple: (psnr_avg, ssim_avg, eval_results)
        """
        # Safety check: if evaluation dataset is empty, return default values
        if len(self.eval_Dataset_ins) == 0:
            print("Warning: evaluation dataset is empty; skipping evaluation. Best model will be selected by training loss.")
            return 0.0, 0.0, []

        # Evaluation mode, disable dropout/fix BN
        self.phs_net.eval()

        with torch.no_grad():
            coefficient = 0
            eval_results = []  # Store detailed results for each image
            eval_loader = DataLoader(self.eval_Dataset_ins,
                                    batch_size=1,
                                    num_workers=8,
                                    pin_memory=True,
                                    prefetch_factor=4  # Number of batches to prefetch per worker
                                    )
            # Iterate batches using DataLoader
            for _, (img, image_ints_ts, image_amp_slm, image_depth_slm, rgb_path) in tqdm(enumerate(eval_loader),
                                                                                         total=len(eval_loader),
                                                                                         desc="Evaluating Batches",
                                                                                         position=1,
                                                                                         leave=True,
                                                                                         dynamic_ncols=True,
                                                                                         miniters=5):
                image_amp_slm = image_amp_slm.to(self.device)
                image_depth_slm = image_depth_slm.to(self.device)
                u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                _u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
                depth_num = self.eval_optics_ins.layer_num

                # Select global phase kernel for backward propagation based on flag
                if self.use_structural_propagation_in_evaluation:
                    h_backward_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                    h_backward_delta_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                else:
                    h_backward_g_selected = self.eval_optics_ins.h_backward_g
                    h_backward_delta_g_selected = self.eval_optics_ins.h_backward_delta_g

                for dep in range(depth_num):
                    if dep != 0:
                        u[image_depth_slm == (depth_num - dep - 1) / depth_num] = _u[
                            image_depth_slm == (depth_num - dep - 1) / depth_num]
                    if dep == depth_num - 1:
                        u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_s, h_backward_g_selected, u0=u)
                    else:
                        u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_delta_s, h_backward_delta_g_selected, u0=u)

                # Complex amplitude
                u = u / torch.max(torch.abs(u))
                # Real and imaginary parts (commented out)
                # ri = torch.cat((torch.real(u), torch.imag(u)), -3)
                # Amplitude and phase (commented out)
                # aa = torch.cat((torch.abs(u), torch.angle(u)), -3)

                phs_slm = self.phs_net(u)
                
                # Generate hologram (phase map)
                max_phs = 2 * np.pi
                output_phase = ((phs_slm - phs_slm.mean() + max_phs / 2) % max_phs) / max_phs
                hologram = ((output_phase[0, 0, ...]) * 255).round().cpu().detach().squeeze().numpy().astype(np.uint8)
                
              
                if Hyperparams.SAVE:
                    phase_out_8bit = hologram
                    eval_output_dir = os.path.join(self.root_path or '.', 'eval_outputs')
                    os.makedirs(eval_output_dir, exist_ok=True)
                    img_phs_path = os.path.join(
                        eval_output_dir,
                        os.path.splitext(os.path.basename(rgb_path[0]))[0] + '_' +
                        os.path.splitext(os.path.basename(self.root_path))[0] + '_phase.png'
                    )
                    cv2.imwrite(img_phs_path, phase_out_8bit)

                phs_slm = phs_slm - self.eval_optics_ins.phase_grating
                # Holographic reconstruction
                rec_amp_slm = torch.zeros_like(image_amp_slm)
                rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))

                # Select global phase kernel for forward propagation based on flag
                if self.use_structural_propagation_in_evaluation:
                    h_forward_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                    h_forward_delta_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                else:
                    h_forward_g_selected = self.eval_optics_ins.h_forward_g
                    h_forward_delta_g_selected = self.eval_optics_ins.h_forward_delta_g

                for dep in range(depth_num):
                    if dep == 0:
                        rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_s, h_forward_g_selected, phs_in=phs_slm)
                    else:
                        rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_delta_s, h_forward_delta_g_selected, u0=rec_u)
                    rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                        image_depth_slm == (depth_num - dep - 1) / depth_num]

                # Linear transformation for image brightness and contrast
                rec_ints = rec_amp_slm ** 2
                coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
                coefficient = coefficient + coefficient_temp
                
                # uint8 version for saving
                rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

                # Only save to D_phs_rec when needed (keep original logic)
                if Hyperparams.SAVE:
                    eval_output_dir = os.path.join(self.root_path or '.', 'eval_outputs')
                    os.makedirs(eval_output_dir, exist_ok=True)
                    img_rec_path = os.path.join(
                        eval_output_dir,
                        os.path.splitext(os.path.basename(rgb_path[0]))[0] + '_' +
                        os.path.splitext(os.path.basename(self.root_path))[0] + '_rec.png'
                    )
                    cv2.imwrite(img_rec_path, rec_img_ints)

                rec = rec_img_ints
                img = img.squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
                psnr_value2 = psnr(img, rec, data_range=255)
                psnr_value3 = psnr_value2
                # Since converted grayscale images are in range [0, 255], data_range is 255.
                # If converted to float and in range [0, 1], data_range should be 1
                ssim_value = ssim(img, rec, data_range=255)
                
                # Get base filename (remove path and extension)
                base_filename = os.path.splitext(os.path.basename(rgb_path[0]))[0]
                
                
                # Store detailed results for each image
                result = {
                    'hologram': hologram,
                    'reconstruction': rec_img_ints,  # uint8 version for saving
                    'metrics': {
                        'psnr': float(psnr_value3),  # PSNR calculated using [0,1] range
                        'ssim': float(ssim_value),
                        'coefficient': float(coefficient_temp.item())
                    },
                    'filename': base_filename
                }
                eval_results.append(result)
                
                # Still use skimage version when saving for consistency, but keep both versions for debugging
                PSNR_SSIM_data = np.array([os.path.splitext(os.path.basename(self.root_path))[0], epoch,
                                           os.path.splitext(os.path.basename(rgb_path[0]))[0], psnr_value2, psnr_value3, ssim_value, coefficient_temp.cpu()])
                self.save_train_data(self.PSD_path, PSNR_SSIM_data)
            coefficient_avg = coefficient / len(self.eval_Dataset_ins)
            PSNR_SSIM_data = self.load_train_data(self.PSD_path)
            LineData = np.zeros((1, 3), dtype='float32')
            LineData[:] = np.sum(PSNR_SSIM_data[epoch * len(self.eval_Dataset_ins):(epoch + 1) * len(self.eval_Dataset_ins), 3:6].astype('float32'),
                axis=0) / len(self.eval_Dataset_ins)
            print("Eval {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient_Avg:{:.4f}".format(
                    os.path.basename(os.path.normpath(self.eval_path)),
                    LineData[0, 1], LineData[0, 2], coefficient_avg.item()))  # Use [0, 1] i.e., psnr_value3
            psnr_avg = LineData[0, 1]  # Use psnr_value3 as average value
            ssim_avg = LineData[0, 2]  # Use calculated SSIM average value

            if Hyperparams.PSNRSSIM:
                Hyperparams.PSNR.append(LineData[0, 1])  # Also use psnr_value3 for consistency
                Hyperparams.SSIM.append(LineData[0, 2])


        
        return psnr_avg, ssim_avg, eval_results  # Return average PSNR, SSIM, and detailed results

    def double_phase_encoding(self, u, reduce_resolution=True, epsilon=1e-8):
        """
        Double Phase Encoding (DPE) function - encode complex amplitude u into pure phase hologram.
        
        Args:
            u (torch.Tensor): Input complex amplitude [B, C, H, W]
            reduce_resolution (bool): Whether to reduce resolution to simulate real DPE effects
            epsilon (float): Small constant to prevent numerical instability
            
        Returns:
            torch.Tensor: DPE-encoded phase hologram [B, C, H, W]
        """
        # Extract amplitude and phase
        A = torch.abs(u)  # Amplitude
        phi = torch.angle(u)  # Phase
        
        # Normalize amplitude to [0, 1]
        A_norm = A / (torch.max(A) + epsilon)
        
        # Prevent acos numerical instability by clamping A_norm to [0, 1] range
        A_norm = torch.clamp(A_norm, 0.0, 1.0 - epsilon)
        
        # Calculate phase offset
        delta_theta = torch.acos(A_norm)
        
        # Calculate two phase channels
        theta1 = phi + delta_theta  # First phase channel
        theta2 = phi - delta_theta  # Second phase channel
        
        if reduce_resolution:
            # Simulate real DPE: reduce resolution then interpolate
            # Halve the original resolution
            B, C, H, W = u.shape
            
            # Downsample to half resolution
            theta1_half = torch.nn.functional.interpolate(theta1, size=(H//2, W//2), mode='bilinear', align_corners=False)
            theta2_half = torch.nn.functional.interpolate(theta2, size=(H//2, W//2), mode='bilinear', align_corners=False)
            
            # Create full-size phase map
            phs_slm_DPE = torch.zeros_like(u, dtype=torch.float32)
            
            # Upsample back to original size and interleave
            theta1_full = torch.nn.functional.interpolate(theta1_half, size=(H, W), mode='bilinear', align_corners=False)
            theta2_full = torch.nn.functional.interpolate(theta2_half, size=(H, W), mode='bilinear', align_corners=False)
            
            # Interleave: fill even columns with theta1, odd columns with theta2
            phs_slm_DPE[:, :, :, ::2] = theta1_full[:, :, :, ::2]    # Even columns
            phs_slm_DPE[:, :, :, 1::2] = theta2_full[:, :, :, 1::2]  # Odd columns
            
        else:
            # Ideal DPE without resolution reduction
            phs_slm_DPE = torch.zeros_like(u, dtype=torch.float32)
            
            # Direct interleaving
            phs_slm_DPE[:, :, :, ::2] = theta1[:, :, :, ::2]    # Even columns
            phs_slm_DPE[:, :, :, 1::2] = theta2[:, :, :, 1::2]  # Odd columns
        
        # Ensure phase range is in [-π, π]
        phs_slm_DPE = torch.fmod(phs_slm_DPE + np.pi, 2 * np.pi) - np.pi
        
        return phs_slm_DPE

    def predict(self, img_path, depth_path, phs_path, rec_path, mul_rec_path, use_structural_propagation=False, save_asm_input_field=False, colorize_asm_input_phase=False, output_dir_for_save='', save_prefix=None):
        """
        Predict phase hologram from RGB and depth images using trained model.
        
        Args:
            img_path: Path to RGB image
            depth_path: Path to depth image
            phs_path: Path to save phase hologram
            rec_path: Path to save reconstruction image
            mul_rec_path: Path to save multi-layer reconstruction (if enabled)
            use_structural_propagation: Whether to use structural propagation mode
            save_asm_input_field: Whether to save ASM input complex field
            colorize_asm_input_phase: Whether to colorize phase map when saving
            output_dir_for_save: Output directory for saved fields
            save_prefix: Optional prefix for auxiliary saved fields
        """
        # Dynamically load model, respecting self.device setting
        map_location = self.device if self.device.type == 'cuda' else 'cpu'
        self.phs_net = load_phase_model(self.model_best_path, device=map_location)
        # Evaluation mode, disable dropout/fix BN
        self.phs_net.eval()
        with torch.no_grad():
            # Verify files exist
            if not os.path.exists(img_path) or not os.path.exists(depth_path):
                raise FileNotFoundError(f"Image {img_path} not found in specified path.")

            depth_num = self.eval_optics_ins.layer_num
            # Channel mapping: Blue=0, Green=1, Red=2
            img = cv2.imread(img_path)[:, :, self.eval_optics_ins.channel]
            if Hyperparams.DIFF:
                diff = cv2.imread(f'diff/diff_{self.eval_optics_ins.channel}.png')[:, :, self.eval_optics_ins.channel].astype(np.float32) / 255 + 0.001
                img = img.astype(np.float32) / diff
                img = img ** (1/Hyperparams.GAMMA[self.eval_optics_ins.channel])
                img = img / img.max() * 255

            # Load depth map without changing original format to check channel number
            image_depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            
            # Check channel number, convert to grayscale if not single channel
            if image_depth_raw.ndim > 2 and image_depth_raw.shape[2] > 1:
                print(f"Warning: depth image {os.path.basename(depth_path)} is multi-channel (shape: {image_depth_raw.shape}); converting to grayscale.")
                image_depth = cv2.cvtColor(image_depth_raw, cv2.COLOR_BGR2GRAY)
            else:
                image_depth = image_depth_raw

            if img.shape[0] != self.eval_optics_ins.LCoS_res_h or img.shape[1] != self.eval_optics_ins.LCoS_res_w:
                img = cv2.resize(img, (self.eval_optics_ins.LCoS_res_w, self.eval_optics_ins.LCoS_res_h))  # Resize
            if image_depth.shape[0] != self.eval_optics_ins.LCoS_res_h or image_depth.shape[1] != self.eval_optics_ins.LCoS_res_w:
                image_depth = cv2.resize(image_depth, (self.eval_optics_ins.LCoS_res_w, self.eval_optics_ins.LCoS_res_h))  # Resize

            image = img.astype(np.float32) / 255
            image_amp = np.sqrt(image)  # Initial light field with phase=0
            image_amp_slm = torch.Tensor(image_amp).unsqueeze(0).unsqueeze(0).to(self.device)  # Add dimensions and move to GPU
            image_ints_ts = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)

            # Depth order issue: 0=black, 255(1)=white. To match human perception, use 1-depth, so 255(1)=black, 0=white
            image_depth = image_depth.astype(np.float32) / 255
            # Match SLM depth, perform depth inversion
            if Hyperparams.DEPTH_INVERSION:
                image_depth = 1 - image_depth
            for dep in range(depth_num):  # Depth discretization
                image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
            image_depth_slm = torch.Tensor(image_depth).unsqueeze(0).unsqueeze(0).to(self.device)
            # Depth ordering: 5-4,4-3,3-2,2-1,1-0;1-2,2-3,3-4,4-5,5-0
            u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
            _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm)*torch.pi*Hyperparams.init_phs)
            

            # Select global phase kernel for backward propagation based on flag
            if use_structural_propagation:
                # When using only structural phase, global phase component is 1 (i.e., phase=0)
                h_backward_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                h_backward_delta_g_selected = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
            else:
                h_backward_g_selected = self.eval_optics_ins.h_backward_g
                h_backward_delta_g_selected = self.eval_optics_ins.h_backward_delta_g

            for depth_index in range(depth_num):
                if depth_index != 0:
                    u[image_depth_slm == (depth_num - depth_index - 1) / depth_num] = _u[
                        image_depth_slm == (depth_num - depth_index - 1) / depth_num]
                if depth_index == depth_num - 1:
                    u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_s, h_backward_g_selected, u0=u)
                else:
                    u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_delta_s, h_backward_delta_g_selected, u0=u)

            u = u / torch.max(torch.abs(u))

            # If flag is True, save ASM-calculated diffraction field
            if save_asm_input_field:
                base_name = save_prefix or os.path.splitext(os.path.basename(img_path))[0]
                self._save_complex_field(u, output_dir_for_save, f"{base_name}_asm_input", colorize=colorize_asm_input_phase)

            if Hyperparams.TIME and self.device.type == "cuda":
                total_time = 0

                # Define CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                for i in range(10):
                    # Record program start time
                    start_event.record()

                    __u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)
                    _u = torch.polar(image_amp_slm, torch.ones_like(image_amp_slm) * torch.pi * Hyperparams.init_phs)
                    
                    for depth_index in range(depth_num):
                        if depth_index != 0:
                            __u[image_depth_slm == (depth_num - depth_index - 1) / depth_num] = _u[
                                image_depth_slm == (depth_num - depth_index - 1) / depth_num
                            ]
                        if depth_index == depth_num - 1:
                            __u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_s, self.eval_optics_ins.h_backward_g, u0=__u)
                        else:
                            __u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_backward_delta_s, self.eval_optics_ins.h_backward_delta_g, u0=__u)

                    # Record program end time
                    end_event.record()

                    # Wait for events to complete and calculate time
                    torch.cuda.synchronize()
                    elapsed_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

                    if i >= 5:
                        total_time += elapsed_time

                # Calculate average time
                average_time = total_time / 5
                Hyperparams.time1.append(average_time)
                print(f'Average elapsed time: {average_time:.4f} seconds')

            # Real and imaginary parts (debug output, commented out)
            # ri = torch.cat((torch.real(u), torch.imag(u)), -3)
            # Amplitude and phase (debug output, commented out)
            # aa = torch.cat((torch.abs(u), torch.angle(u)), -3)
            # Output intensity representation of real and imaginary parts
            Real_Imag = 0
            if Real_Imag == 1:
                # Real part
                real = torch.real(u)
                real_out = (real * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Real.png', real_out)
                # Imaginary part
                imag = torch.imag(u)
                imag_out = (imag * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Imag.png', imag_out)
                        
            Amp_Phs = 0
            if Amp_Phs == 1:
                # Amplitude
                amp = torch.abs(u)
                amp_out = (amp * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Amp.png', amp_out)
                # Phase
                phs = torch.angle(u)
                phs_out = (phs * 255).round().cpu().detach().squeeze().squeeze().numpy().astype(np.uint8)
                cv2.imwrite(r'draw/Test_Phs.png', phs_out)

            phs_slm = self.phs_net(u)

            if Hyperparams.TIME and self.device.type == "cuda":
                total_time = 0
                for i in range(35):
                    # Record program start time
                    torch.cuda.synchronize()
                    start_time = time.time()

                    _ = self.phs_net(u)

                    # Record program end time
                    torch.cuda.synchronize()
                    end_time = time.time()
                    # Calculate program runtime
                    elapsed_time = end_time - start_time
                    if i >= 5:
                        total_time = total_time + elapsed_time
                Hyperparams.time2.append(total_time / 30)

            if Hyperparams.SAVE:
                max_phs = 2 * np.pi
                output_phase = ((phs_slm - phs_slm.mean() + max_phs / 2) % max_phs) / max_phs
                phase_out_8bit = ((output_phase[0, 0, ...]) * 255).round().cpu().detach().squeeze().numpy().astype(
                    np.uint8)
                # Ensure directory exists
                os.makedirs(os.path.dirname(phs_path), exist_ok=True)
                cv2.imwrite(phs_path, phase_out_8bit)

            phs_slm = phs_slm - self.eval_optics_ins.phase_grating

            # phs_slm = self.double_phase_encoding(u)    # Double phase encoding
            # Holographic reconstruction
            rec_amp_slm = torch.zeros_like(image_amp_slm)
            rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))

            # Select global phase component based on flag
            if use_structural_propagation:
                # To disable global phase, we multiply by 1 (i.e., phase=0), not 0.
                h_g = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
                h_delta_g = torch.tensor(1.0, dtype=torch.complex64).to(self.device)
            else:
                h_g = self.eval_optics_ins.h_forward_g
                h_delta_g = self.eval_optics_ins.h_forward_delta_g

            for dep in range(depth_num):
                if dep == 0:
                    rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_s, h_g, phs_in=phs_slm)
                else:
                    rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_delta_s, h_delta_g, u0=rec_u)
                rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                    image_depth_slm == (depth_num - dep - 1) / depth_num]

                if Hyperparams.MUL_SAVE:
                    if mul_rec_path:
                        mul_rec_path_temp = mul_rec_path.replace('.png', f"{str(dep)}.png")
                        mul_rec_amp_slm = torch.abs(rec_u)
                        # Linear transformation for image brightness and contrast
                        mul_rec_ints = mul_rec_amp_slm * mul_rec_amp_slm
                        mul_rec_ints *= torch.sum(image_ints_ts) / torch.sum(mul_rec_ints)
                        mul_rec_ints = (mul_rec_ints * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(
                            np.uint8)
                        # Save reconstruction image
                        cv2.imwrite(mul_rec_path_temp, mul_rec_ints)

            # Linear transformation for image brightness and contrast
            rec_ints = rec_amp_slm ** 2

            coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
            rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(
                0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
            # Alternative reconstruction (commented out): rec_img_ints = (rec_ints*255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)
            if Hyperparams.SAVE:
                # Ensure directory exists
                os.makedirs(os.path.dirname(rec_path), exist_ok=True)
                cv2.imwrite(rec_path, rec_img_ints)

            rec = rec_img_ints

            psnr_value2 = psnr(img, rec, data_range=255)
            psnr_value3 = psnr_value2
            # Since converted grayscale images are in range [0, 255], data_range is 255.
            # If converted to float and in range [0, 1], data_range should be 1
            ssim_value = ssim(img, rec, data_range=255)
            print("Test Image {} | Dataset {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient:{:.4f}".format(
                os.path.splitext(os.path.basename(img_path))[0],
                os.path.splitext(os.path.basename(self.root_path))[0],
                psnr_value3, ssim_value, coefficient_temp.item()))


    def phs_predict(self, rgb_path, depth_path, phs_path, img_rec_path='', phase_grating=True):
        """
        Predict reconstruction from existing phase hologram.
        
        Unlike predict(), this method directly loads an existing phase hologram instead of generating it.
        Used to evaluate quality of phase holograms generated by other methods.
        
        Args:
            rgb_path: Path to RGB image (for comparison)
            depth_path: Path to depth image
            phs_path: Path to existing phase hologram
            img_rec_path: Path to save reconstruction image
            phase_grating: Whether to subtract phase grating
        """
        # Verify files exist
        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            raise FileNotFoundError(f"Image {rgb_path} not found in specified path.")

        depth_num = self.eval_optics_ins.layer_num
        # Channel mapping: Blue=0, Green=1, Red=2
        img = cv2.imread(rgb_path)[:, :, self.eval_optics_ins.channel]
        image_depth = cv2.imread(depth_path, 0)

        # Resize to match SLM resolution
        if img.shape[0] != self.eval_optics_ins.LCoS_res_h or img.shape[1] != self.eval_optics_ins.LCoS_res_w:
            img = cv2.resize(img, (self.eval_optics_ins.LCoS_res_w, self.eval_optics_ins.LCoS_res_h))
        if image_depth.shape[0] != self.eval_optics_ins.LCoS_res_h or image_depth.shape[1] != self.eval_optics_ins.LCoS_res_w:
            image_depth = cv2.resize(image_depth, (self.eval_optics_ins.LCoS_res_w, self.eval_optics_ins.LCoS_res_h))

        image = img.astype(np.float32) / 255
        image_ints_ts = torch.Tensor(image).unsqueeze(0).unsqueeze(0).to(self.device)
        image_amp = np.sqrt(image)  # Initial light field with phase=0
        image_amp_slm = torch.Tensor(image_amp).unsqueeze(0).unsqueeze(0).to(self.device)  # Add dimensions and move to GPU

        # Depth order issue: 0=black, 255(1)=white. To match human perception, use 1-depth, so 255(1)=black, 0=white
        image_depth = image_depth.astype(np.float32) / 255
        for dep in range(depth_num):  # Depth discretization
            image_depth[(image_depth >= dep / depth_num) & (image_depth <= (dep + 1) / depth_num)] = dep / depth_num
        image_depth_slm = torch.Tensor(image_depth).unsqueeze(0).unsqueeze(0).to(self.device)

        # Load existing phase hologram directly (no model inference)
        image_phs_slm = cv2.imread(phs_path)[:, :, self.eval_optics_ins.channel].astype(np.float32) / 255 * 2 * np.pi
        phs_slm = torch.Tensor(image_phs_slm).unsqueeze(0).unsqueeze(0).to(self.device)  # Add dimensions and move to GPU
        if phase_grating:
            phs_slm = phs_slm - self.eval_optics_ins.phase_grating
        # Holographic reconstruction
        rec_amp_slm = torch.zeros_like(image_amp_slm)
        rec_u = torch.polar(image_amp_slm, torch.zeros_like(image_amp_slm))
        for dep in range(depth_num):
            if dep == 0:
                rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_s, self.eval_optics_ins.h_forward_g, phs_in=phs_slm)
            else:
                rec_u = self.eval_optics_ins.prop_asm(self.eval_optics_ins.h_forward_delta_s, self.eval_optics_ins.h_forward_delta_g, u0=rec_u)
            rec_amp_slm[image_depth_slm == (depth_num - dep - 1) / depth_num] = torch.abs(rec_u)[
                image_depth_slm == (depth_num - dep - 1) / depth_num]

            if Hyperparams.MUL_SAVE:
                img_mul_rec_path = r'mul_pred/' + os.path.splitext(os.path.basename(rgb_path))[0] + '_' + \
                            os.path.splitext(os.path.basename(self.root_path))[0] + '_' + \
                            str(self.eval_optics_ins.channel) + '_' + \
                            'rec' + '_' + \
                            str(dep) + '.png'  # Save path
                mul_rec_amp_slm = torch.abs(rec_u)
                # Linear transformation for image brightness and contrast
                mul_rec_ints = mul_rec_amp_slm * mul_rec_amp_slm
                mul_rec_ints *= torch.sum(image_ints_ts) / torch.sum(mul_rec_ints)
                mul_rec_ints = (mul_rec_ints * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(
                    np.uint8)
                # Save reconstruction image
                cv2.imwrite(img_mul_rec_path, mul_rec_ints)

        # Linear transformation for image brightness and contrast
        rec_ints = rec_amp_slm ** 2

        coefficient_temp = torch.sum(image_ints_ts) / torch.sum(rec_ints)
        rec_img_ints = (rec_ints * coefficient_temp * 255).round().squeeze(0).squeeze(0).cpu().detach().numpy().clip(0, 255).astype(np.uint8)

        if Hyperparams.SAVE:
            if img_rec_path=='':
                img_rec_path = r'pred/' + os.path.splitext(os.path.basename(rgb_path))[0] + '_' + \
                            os.path.splitext(os.path.basename(self.root_path))[0] + '_' + \
                            str(self.eval_optics_ins.channel) + '_' + \
                            'rec.png'  # Save path
            cv2.imwrite(img_rec_path, rec_img_ints)

        rec = rec_img_ints

        psnr_value2 = psnr(img, rec, data_range=255)
        psnr_value3 = psnr_value2
        # Since converted grayscale images are in range [0, 255], data_range is 255.
        # If converted to float and in range [0, 1], data_range should be 1
        ssim_value = ssim(img, rec, data_range=255)
        print("Test Image {} | Dataset {} : PSNR: {:.4f}, SSIM: {:.4f}, Coefficient:{:.4f}".format(
            os.path.splitext(os.path.basename(rgb_path))[0],
            os.path.splitext(os.path.basename(self.root_path))[0],
            psnr_value3, ssim_value, coefficient_temp.item()))
            

    

    

        

    @staticmethod
    def save_train_data(data_path, new_data):
        """
        Save training data to file (append mode).
        
        If file exists, loads existing data and appends new row.
        If file doesn't exist, creates new array with new_data.
        
        Args:
            data_path: Path to save data file
            new_data: New data row to append
        """
        filename = data_path
        # Use numpy save function to save data
        # Note: If file already exists, we need to load existing data first, then append new row
        try:
            # Load existing data
            train_data = np.load(filename, allow_pickle=True)
            # Append new row
            train_data = np.vstack((train_data, new_data))
        except FileNotFoundError:
            # If file doesn't exist, create a new 2D array
            train_data = np.array([new_data])
        # Save updated data
        np.save(filename, train_data)

    @staticmethod
    def load_train_data(data_path):
        """
        Load training data from file.
        
        Args:
            data_path: Path to data file
            
        Returns:
            np.ndarray: Loaded training data
        """
        filename = data_path
        # Use numpy load function to read data
        return np.load(filename, allow_pickle=True)

    @staticmethod
    def delete_train_data(data_path):
        """
        Delete training data file.
        
        Args:
            data_path: Path to data file to delete
        """
        filename = data_path
        if os.path.exists(filename):  # Check if file exists
            os.remove(filename)  # Delete file
            print(data_path + "Train data file removed successfully!")
        else:
            print(data_path + "Train data file does not exist")

    @staticmethod
    def _save_complex_field(u: torch.Tensor, out_dir: str, basename: str, colorize: bool = False):
        """Helper to save complex field's amplitude and phase as PNG images."""
        os.makedirs(out_dir, exist_ok=True)
        # Ensure tensor is on CPU and converted to numpy
        amp = torch.abs(u).squeeze().cpu().numpy()
        phs = torch.angle(u).squeeze().cpu().numpy()

        # Normalize amplitude [0, 1] -> [0, 255]
        amp_norm = np.clip(amp, 0.0, 1.0)
        amp_8bit = (amp_norm * 255).astype(np.uint8)

        # Normalize phase [-pi, pi] -> [0, 1] -> [0, 255]
        phs_norm = (phs + np.pi) / (2 * np.pi)
        phs_8bit = (phs_norm * 255).astype(np.uint8)

        cv2.imwrite(os.path.join(out_dir, f'{basename}_amp.png'), amp_8bit)
        
        # Decide whether to colorize phase map based on flag
        phase_path = os.path.join(out_dir, f'{basename}_phs.png')
        if colorize:
            plt.imsave(phase_path, phs_8bit, cmap='Set3')
        else:
            cv2.imwrite(phase_path, phs_8bit)
            
        print(f"Saved ASM input field: {out_dir}/{basename}_amp.png and {basename}_phs.png")


