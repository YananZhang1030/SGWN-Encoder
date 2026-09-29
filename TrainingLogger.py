import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from datetime import datetime
from typing import Dict, List, Optional, Any
import platform
import sys
from hyperparams import Hyperparams

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # Support Chinese font display (legacy)
plt.rcParams['axes.unicode_minus'] = False

class TrainingLogger:
    """
    Training process monitoring and result saving manager.
    
    Handles logging, visualization, and saving of training metrics, configurations,
    and best results during model training.
    """
    
    def __init__(self, root_path: str, experiment_name: str):
        """
        Initialize TrainingLogger.
        
        Args:
            root_path: Root directory for saving all logs and results
            experiment_name: Name of the experiment
        """
        self.root_path = root_path
        self.experiment_name = experiment_name
        
        # Create folder structure
        self.setup_directories()
        
        # Training log
        self.training_log = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'best_psnr': 0.0,
            'best_epoch': -1,
            'hyperparameters': {},
            'config_saved': False
        }
        
        # Best results record
        self.best_metrics = {
            'psnr': 0.0,
            'ssim': 0.0,
            'coefficient_avg': 0.0,
            'epoch': -1,
            'loss': float('inf')
        }
        
    def setup_directories(self):
        """
        Create complete folder structure for organizing training outputs.
        """
        self.dirs = {
            'models': os.path.join(self.root_path, 'models'),
            'training_monitor': os.path.join(self.root_path, 'training_monitor'),
            'best_results': os.path.join(self.root_path, 'best_results'),
            'dataset_free_samples': os.path.join(self.root_path, 'dataset_free_samples'),
            'config_snapshot': os.path.join(self.root_path, 'config_snapshot')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
    
    def save_config_snapshot(self, optics_ins, train_path, eval_path, model):
        """
        Save complete training configuration snapshot.
        
        Saves hyperparameters, model configuration, optics configuration,
        dataset configuration, and environment information.
        
        Args:
            optics_ins: Optics instance
            train_path: Training dataset path
            eval_path: Evaluation dataset path
            model: Model instance
        """
        try:
            # 1. Save hyperparameters
            hyperparams = {}
            try:
                # Safely get configuration values and convert to serializable types
                def safe_convert(value):
                    if hasattr(value, 'item'):  # numpy scalar
                        return value.item()
                    elif hasattr(value, 'tolist'):  # numpy array
                        return value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        return float(value)
                    elif torch.is_tensor(value):
                        return str(value)
                    else:
                        return value
                
                hyperparams.update({
                    'batch_size': safe_convert(getattr(Hyperparams, 'batch_size', 1)),
                    'epochs': safe_convert(getattr(Hyperparams, 'epochs', 20)),
                    'train_image_num': safe_convert(getattr(Hyperparams, 'train_image_num', 100)),
                    'learning_rate': safe_convert(getattr(Hyperparams, 'learning_rate', 0.001)),
                    'learning_rate_step_size': safe_convert(getattr(Hyperparams, 'learning_rate_step_size', 400)),
                    'learning_rate_gamma': safe_convert(getattr(Hyperparams, 'learning_rate_gamma', 0.5)),
                    'optimizer': safe_convert(getattr(Hyperparams, 'optimizer', 'Adam')),
                    'TEST_MODE': safe_convert(getattr(Hyperparams, 'TEST_MODE', False)),
                    'SAVE': safe_convert(getattr(Hyperparams, 'SAVE', True)),
                    'device': str(getattr(Hyperparams, 'device', 'cpu'))
                })
            except Exception as e:
                print(f"Warning: hyperparameter read failed: {e}")
                hyperparams = {'error': 'hyperparams_access_failed'}
            
            with open(os.path.join(self.dirs['config_snapshot'], 'hyperparams.json'), 'w') as f:
                json.dump(hyperparams, f, indent=2)
            
            # 2. Save model configuration
            model_config = {}
            try:
                model_name = 'Unknown'
                param_count = 0
                
                if hasattr(model, '__class__'):
                    model_name = model.__class__.__name__
                elif hasattr(model, '_modules'):
                    # For nn.Sequential, try to get the first module's name
                    modules = list(model._modules.values()) if hasattr(model, '_modules') else []
                    if modules:
                        model_name = modules[0].__class__.__name__
                
                if hasattr(model, 'parameters'):
                    try:
                        param_count = sum(p.numel() for p in model.parameters())
                    except:
                        param_count = 0
                        
                model_config = {
                    'model_name': model_name,
                    'model_parameters_count': param_count,
                    'phase_range': [-3.14159, 3.14159],
                    'activation': 'Hardtanh',
                    'model_type': str(type(model))
                }
            except Exception as e:
                print(f"Warning: model configuration read failed: {e}")
                model_config = {'error': 'model_access_failed'}
            
            with open(os.path.join(self.dirs['config_snapshot'], 'model_config.json'), 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # 3. Save optics configuration
            optics_config = {}
            try:
                optics_config = {
                    'channel': safe_convert(getattr(optics_ins, 'channel', 'unknown')),
                    'layer_num': safe_convert(getattr(optics_ins, 'layer_num', 'unknown')),
                    'LCoS_res_h': safe_convert(getattr(optics_ins, 'LCoS_res_h', 'unknown')),
                    'LCoS_res_w': safe_convert(getattr(optics_ins, 'LCoS_res_w', 'unknown')),
                    'pixel_size': safe_convert(getattr(optics_ins, 'pixel_size', 'not_available')),
                    'wavelength': safe_convert(getattr(optics_ins, 'wavelength', 'not_available'))
                }
            except Exception as e:
                print(f"Warning: optics configuration read failed: {e}")
                optics_config = {'error': 'optics_access_failed'}
            
            with open(os.path.join(self.dirs['config_snapshot'], 'optics_config.json'), 'w') as f:
                json.dump(optics_config, f, indent=2)
            
            # 4. Save dataset configuration
            dataset_config = {
                'train_dataset': str(train_path),
                'eval_dataset': str(eval_path),
                'train_data_type': 'dataset_free_random' if 'DF_D' in str(train_path) else 'file_based',
                'eval_data_type': 'RGBD_real_data'
            }
            
            with open(os.path.join(self.dirs['config_snapshot'], 'dataset_config.json'), 'w') as f:
                json.dump(dataset_config, f, indent=2)
            
            # 5. Save runtime environment information
            environment_info = {}
            try:
                environment_info = {
                    'timestamp': datetime.now().isoformat(),
                    'python_version': sys.version,
                    'pytorch_version': torch.__version__ if 'torch' in globals() else 'unknown',
                    'cuda_available': torch.cuda.is_available() if 'torch' in globals() else False,
                    'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                    'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'current_device': str(getattr(Hyperparams, 'device', 'unknown')),
                    'system_info': {
                        'platform': platform.platform(),
                        'system': platform.system(),
                        'processor': platform.processor(),
                        'python_implementation': platform.python_implementation()
                    }
                }
            except Exception as e:
                print(f"Warning: environment information read failed: {e}")
                environment_info = {'error': 'environment_access_failed', 'timestamp': datetime.now().isoformat()}
            
            with open(os.path.join(self.dirs['config_snapshot'], 'environment_info.json'), 'w') as f:
                json.dump(environment_info, f, indent=2)
            
            self.training_log['config_saved'] = True
            
        except Exception as e:
            print(f"Warning: configuration snapshot save failed: {e}")
            print(f"Error details: {type(e).__name__}: {str(e)}")
            # Continue training even if failed, don't affect main functionality
            self.training_log['config_saved'] = False
    
    def log_epoch(self, epoch: int, train_loss: float, coefficient_avg: float, 
                  psnr: float, ssim: float, training_time: float, eval_available: bool = True):
        """
        Log training data for each epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            coefficient_avg: Average energy coefficient
            psnr: Peak Signal-to-Noise Ratio
            ssim: Structural Similarity Index
            training_time: Time taken for this epoch
            eval_available: Whether evaluation metrics were produced this epoch;
                when False, the best epoch is selected by training loss instead
            
        Returns:
            bool: True if this is the best result so far, False otherwise
        """
        epoch_data = {
            'epoch': epoch,
            'train_loss': float(train_loss),
            'coefficient_avg': float(coefficient_avg),
            'psnr': float(psnr),
            'ssim': float(ssim),
            'training_time': float(training_time),
            'timestamp': datetime.now().isoformat()
        }
        
        self.training_log['epochs'].append(epoch_data)
        
        # Check if this is the best result
        is_best = False
        if eval_available:
            if psnr > self.best_metrics['psnr']:
                self.best_metrics.update({
                    'psnr': float(psnr),
                    'ssim': float(ssim),
                    'coefficient_avg': float(coefficient_avg),
                    'epoch': epoch,
                    'loss': float(train_loss)
                })
                self.training_log['best_psnr'] = float(psnr)
                self.training_log['best_epoch'] = epoch
                is_best = True
        elif float(train_loss) < self.best_metrics['loss']:
            # No evaluation data: fall back to training loss for best-model selection
            self.best_metrics.update({
                'psnr': 0.0,
                'ssim': 0.0,
                'coefficient_avg': float(coefficient_avg),
                'epoch': epoch,
                'loss': float(train_loss)
            })
            self.training_log['best_epoch'] = epoch
            is_best = True
        
        return is_best
    
    def save_model(self, model, epoch: int, is_best: bool = False, is_latest: bool = True):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch number
            is_best: Whether this is the best model
            is_latest: Whether to save as latest model
        """
        try:
            if is_best:
                model_path = os.path.join(self.dirs['models'], 'model_state_dict.pt')
                torch.save(model.state_dict(), model_path)
                
            if is_latest:
                model_path = os.path.join(self.dirs['models'], 'latest_model_state_dict.pt')
                torch.save(model.state_dict(), model_path)
                
        except Exception as e:
            print(f"Warning: model save failed: {e}")
    
    def save_best_results(self, hologram: np.ndarray, reconstruction: np.ndarray, 
                         metrics: Dict[str, float]):
        """
        Save best results.
        
        Args:
            hologram: Hologram (phase map) image
            reconstruction: Reconstructed image
            metrics: Evaluation metrics dictionary
        """
        try:
            # Save images
            cv2.imwrite(os.path.join(self.dirs['best_results'], 'hologram.png'), hologram)
            cv2.imwrite(os.path.join(self.dirs['best_results'], 'reconstruction.png'), reconstruction)
            
            # Save metrics
            with open(os.path.join(self.dirs['best_results'], 'metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
                
            
        except Exception as e:
            print(f"Warning: best results save failed: {e}")
    
    def save_dataset_free_samples(self, samples: List[Dict[str, np.ndarray]], 
                        sample_info: Optional[Dict] = None):
        """
        Save dataset-free training sample visualizations.
        
        Args:
            samples: List of sample dictionaries containing real, imag, amp, phase arrays
            sample_info: Optional metadata about samples
        """
        try:
            for i, sample in enumerate(samples[:5]):  # Only save first 5 samples
                sample_id = f"{i+1:03d}"
                
                if 'real' in sample:
                    cv2.imwrite(os.path.join(self.dirs['dataset_free_samples'], 
                               f'sample_{sample_id}_real.png'), sample['real'])
                if 'imag' in sample:
                    cv2.imwrite(os.path.join(self.dirs['dataset_free_samples'], 
                               f'sample_{sample_id}_imag.png'), sample['imag'])
                if 'amp' in sample:
                    cv2.imwrite(os.path.join(self.dirs['dataset_free_samples'], 
                               f'sample_{sample_id}_amp.png'), sample['amp'])
                if 'phase' in sample:
                    cv2.imwrite(os.path.join(self.dirs['dataset_free_samples'], 
                               f'sample_{sample_id}_phase.png'), sample['phase'])
            
            # Save sample information
            if sample_info:
                with open(os.path.join(self.dirs['dataset_free_samples'], 'sample_info.json'), 'w') as f:
                    json.dump(sample_info, f, indent=2)
                    
            
        except Exception as e:
            print(f"Warning: dataset-free sample save failed: {e}")
    
    def save_evaluation_results(self, eval_results: List[Dict[str, Any]]):
        """
        Save evaluation results.
        
        Args:
            eval_results: List of evaluation result dictionaries
        """
        try:
            metrics_summary = []
            
            for i, result in enumerate(eval_results):
                result_id = f"{i+1:03d}"
                
                # Save images
                if 'hologram' in result:
                    cv2.imwrite(os.path.join(self.dirs['eval_gallery'], 
                               f'eval_{result_id}_hologram.png'), result['hologram'])
                if 'reconstruction' in result:
                    cv2.imwrite(os.path.join(self.dirs['eval_gallery'], 
                               f'eval_{result_id}_reconstruction.png'), result['reconstruction'])
                
                # Collect metrics
                if 'metrics' in result:
                    metrics_summary.append({
                        'eval_id': result_id,
                        'filename': result.get('filename', f'eval_{result_id}'),
                        **result['metrics']
                    })
            
            # Save evaluation metrics summary
            with open(os.path.join(self.dirs['eval_gallery'], 'eval_metrics.json'), 'w') as f:
                json.dump(metrics_summary, f, indent=2)
                
            
        except Exception as e:
            print(f"Warning: evaluation result save failed: {e}")
    
    def plot_training_curves(self):
        """
        Plot training curves.
        
        Generates multiple plots: combined view, loss curve, and PSNR+SSIM curves.
        """
        if len(self.training_log['epochs']) < 2:
            return
        
        try:
            epochs = [e['epoch'] for e in self.training_log['epochs']]
            losses = [e['train_loss'] for e in self.training_log['epochs']]
            psnrs = [e['psnr'] for e in self.training_log['epochs']]
            ssims = [e['ssim'] for e in self.training_log['epochs']]
            coefficients = [e['coefficient_avg'] for e in self.training_log['epochs']]
            
            # Create subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Training Progress - {self.experiment_name}', fontsize=16)
            
            # Loss curve
            ax1.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # PSNR curve
            ax2.plot(epochs, psnrs, 'r-', linewidth=2, label='PSNR')
            ax2.axhline(y=max(psnrs), color='r', linestyle='--', alpha=0.7, 
                       label=f'Best: {max(psnrs):.2f} dB')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('PSNR')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # SSIM curve
            ax3.plot(epochs, ssims, 'g-', linewidth=2, label='SSIM')
            ax3.axhline(y=max(ssims), color='g', linestyle='--', alpha=0.7, 
                       label=f'Best: {max(ssims):.3f}')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('SSIM')
            ax3.set_title('SSIM')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Coefficient curve
            ax4.plot(epochs, coefficients, 'm-', linewidth=2, label='Coefficient Avg')
            ax4.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Ideal (1.0)')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Energy Loss Coefficient')
            ax4.set_title('Energy Loss Coefficient')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
            
            plt.tight_layout()
            
            # Save figure
            plt.savefig(os.path.join(self.dirs['training_monitor'], 'training_curves.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save Loss curve separately
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, losses, 'b-', linewidth=2, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training Loss - {self.experiment_name}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(self.dirs['training_monitor'], 'loss_curve.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save PSNR+SSIM curves separately
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ax1.plot(epochs, psnrs, 'r-', linewidth=2, label='PSNR')
            ax1.axhline(y=max(psnrs), color='r', linestyle='--', alpha=0.7)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('PSNR (dB)')
            ax1.set_title('PSNR Progress')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.plot(epochs, ssims, 'g-', linewidth=2, label='SSIM')
            ax2.axhline(y=max(ssims), color='g', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('SSIM')
            ax2.set_title('SSIM Progress')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.dirs['training_monitor'], 'psnr_ssim_curve.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Warning: training curve plotting failed: {e}")
    
    def save_training_log(self):
        """
        Save training log.
        
        Saves JSON format detailed log and human-readable text summary.
        Also generates training curves.
        """
        try:
            # Update end time
            self.training_log['end_time'] = datetime.now().isoformat()
            
            # Save detailed log in JSON format
            with open(os.path.join(self.dirs['training_monitor'], 'training_log.json'), 'w') as f:
                json.dump(self.training_log, f, indent=2)
            
            # Save human-readable text summary
            self.save_training_summary()
            
            # Plot training curves
            self.plot_training_curves()
            
            
        except Exception as e:
            print(f"Warning: training log save failed: {e}")
    
    def save_training_summary(self):
        """
        Save training summary.
        
        Creates a human-readable text file with experiment summary information.
        """
        try:
            summary_path = os.path.join(self.dirs['training_monitor'], 'summary.txt')
            
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"Training Summary - {self.experiment_name}\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Experiment Name: {self.training_log['experiment_name']}\n")
                f.write(f"Start Time: {self.training_log.get('start_time', 'N/A')}\n")
                f.write(f"End Time: {self.training_log.get('end_time', 'N/A')}\n\n")
                
                f.write("Best Results:\n")
                f.write(f"  Best PSNR: {self.best_metrics['psnr']:.4f} dB (Epoch {self.best_metrics['epoch']})\n")
                f.write(f"  Corresponding SSIM: {self.best_metrics['ssim']:.4f}\n")
                f.write(f"  Corresponding Loss: {self.best_metrics['loss']:.4f}\n")
                f.write(f"  Corresponding Coefficient: {self.best_metrics['coefficient_avg']:.4f}\n\n")
                
                if self.training_log['epochs']:
                    final_epoch = self.training_log['epochs'][-1]
                    f.write("Final Results:\n")
                    f.write(f"  Final Epoch PSNR: {final_epoch['psnr']:.4f} dB\n")
                    f.write(f"  Final Epoch SSIM: {final_epoch['ssim']:.4f}\n")
                    f.write(f"  Final Epoch Loss: {final_epoch['train_loss']:.4f}\n\n")
                    
                    total_time = sum(e['training_time'] for e in self.training_log['epochs'])
                    f.write(f"Total Training Time: {total_time:.2f} seconds\n")
                    f.write(f"Average Time per Epoch: {total_time/len(self.training_log['epochs']):.2f} seconds\n")
                    
        except Exception as e:
            print(f"Warning: training summary save failed: {e}")

    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get experiment summary information.
        
        Returns:
            dict: Summary dictionary containing experiment name, best metrics, total epochs, and directories
        """
        return {
            'experiment_name': self.experiment_name,
            'best_metrics': self.best_metrics,
            'total_epochs': len(self.training_log['epochs']),
            'directories': self.dirs
        }
