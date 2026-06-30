import numpy as np
import torch
import torch.fft
import torch.nn as nn
from hyperparams import Hyperparams

class Optics:
    def __init__(self, channel, first_z, delta_z, layer_num, factor=0.75, grating_type='vertical', 
                 LCoS_res_h=2160, LCoS_res_w=3840, LCoS_pitch=3.6e-6, linear_conv=True, band_limit=True):
        # Wavelength dictionary, key is channel, value is corresponding wavelength
        self.wavelength_dict = {
            0: np.array(4.5e-7),    # 'B'
            1: np.array(5.2e-7),    # 'G'
            2: np.array(6.38e-7)    # 'R'
        }
        self.channel = channel
        self.wavelength = self.wavelength_dict[channel]
        self.linear_conv = linear_conv
        self.band_limit = band_limit
        self.factor = factor
        self.device = Hyperparams.device
        

        self.first_z = first_z
        self.delta_z = delta_z
        
        self.layer_num = layer_num
        self.LCoS_res_h = LCoS_res_h
        self.LCoS_res_w = LCoS_res_w
        self.LCoS_pitch = LCoS_pitch
        
        self.LCoS_LC_h = self.LCoS_res_h * self.LCoS_pitch
        self.LCoS_LC_w = self.LCoS_res_w * self.LCoS_pitch
        self.dtype_t = torch.float32
        self.dtype_n = np.float32
        
        self.phase_grating = self.phase_grating_init(grating_type=grating_type)
        
        # Pre-calculate transfer function (separated into spatial term 's' and global term 'g')
        # ASM backward transfer function
        self.h_backward_delta_s, self.h_backward_delta_g = self.precal_h(prop_z=self.delta_z)
        backward_z = -(self.first_z + self.delta_z * (self.layer_num - 1))
        self.h_backward_s, self.h_backward_g = self.precal_h(prop_z=backward_z)
        
        # ASM first forward transfer function
        self.h_forward_s, self.h_forward_g = self.precal_h(prop_z=self.first_z, factor=self.factor)
        self.h_forward_df_s, self.h_forward_df_g = self.precal_h(prop_z=self.first_z, factor=self.factor)
        
        # ASM delta forward transfer function
        self.h_forward_delta_s, self.h_forward_delta_g = self.precal_h(prop_z=self.delta_z, factor=self.factor)



    def phase_grating_init(self, grating_type='vertical'):
        """
        Initialize phase grating, supporting vertical, horizontal, and 2D gratings.

        Args:
            grating_type (str): Grating type, options include 'vertical', 'horizontal', '2d'.
                                'vertical' for vertical grating,
                                'horizontal' for horizontal grating,
                                '2d' for 2D grating.
        Returns:
            row_D_phase_grating (torch.Tensor): Initialized grating.
        """
        # Create a phase matrix with all zeros
        phase_grating = torch.zeros((1, 1, self.LCoS_res_h, self.LCoS_res_w)).to(self.device)

        if grating_type == 'vertical':
            # Vertical grating: generate grating in width direction (set π phase for every other column)
            phase_grating[..., 0::2, :] = np.pi
        elif grating_type == 'vertical4':
            # Vertical grating variant: generate grating in width direction (set π phase for every 3rd and 4th column)
            phase_grating[..., 0::3, :] = np.pi
            phase_grating[..., 0::4, :] = np.pi
        elif grating_type == 'horizontal':
            # Horizontal grating: generate grating in height direction (set π phase for every other row)
            phase_grating[..., :, 0::2] = np.pi
        elif grating_type == 'horizontal4':
            # Horizontal grating variant: generate grating in height direction (set π phase for every 3rd and 4th row)
            phase_grating[..., :, 0::3] = np.pi
            phase_grating[..., :, 0::4] = np.pi
        elif grating_type == '2d':
            # 2D grating: generate grating in both height and width directions (checkerboard pattern)
            phase_grating[..., 0::2, 0::2] = np.pi
            phase_grating[..., 1::2, 1::2] = np.pi
        elif grating_type == '2d4':
            # 2D grating variant: generate grating in both directions (checkerboard pattern with 3rd and 4th pattern)
            phase_grating[..., 0::3, 0::3] = np.pi
            phase_grating[..., 0::4, 0::4] = np.pi
            phase_grating[..., 1::3, 1::3] = np.pi
            phase_grating[..., 1::4, 1::4] = np.pi
        else:
            raise ValueError("Invalid grating_type. Choose from 'vertical', 'horizontal', or '2d'.")
        
        return phase_grating


    def precal_h(self, prop_z, factor=1.0):
        # Linear convolution
        num_y = self.LCoS_res_h * 2 if self.linear_conv else self.LCoS_res_h
        num_x = self.LCoS_res_w * 2 if self.linear_conv else self.LCoS_res_w

        y, x = (self.LCoS_pitch * float(num_y), self.LCoS_pitch * float(num_x))

        fy = np.linspace(-1 / (2 * self.LCoS_pitch) + 0.5 / (2 * y), 1 / (2 * self.LCoS_pitch) - 0.5 / (2 * y), num_y).astype(self.dtype_n)
        fx = np.linspace(-1 / (2 * self.LCoS_pitch) + 0.5 / (2 * x), 1 / (2 * self.LCoS_pitch) - 0.5 / (2 * x), num_x).astype(self.dtype_n)
        FX, FY = np.meshgrid(fx, fy)

        # Core modification: separate global phase and spatial phase
        k = 2 * np.pi / self.wavelength
        
        # 1. Global Phase Term
        # Phase offset caused by propagation, independent of spatial frequency (fx, fy)
        global_phase_term_Hg = torch.exp(torch.tensor(1j * k * prop_z, dtype=torch.complex64)).to(self.device)

        # 2. Spatial/Structural Phase Term
        # Phase change dependent on spatial frequency (fx, fy), which determines the structure of the diffraction pattern
        term_inside_sqrt = 1.0 - (self.wavelength * FX)**2 - (self.wavelength * FY)**2
        
        # Handle evanescent waves, avoid taking square root of negative numbers
        # When term_inside_sqrt < 0, sqrt will produce imaginary numbers, exp(j*k*z*j*...) = exp(-k*z*...) becomes a decay term
        # We directly set its phase contribution to 0 here, as they decay in the far field
        phase_spatial_numpy = k * prop_z * (np.sqrt(np.maximum(0, term_inside_sqrt)) - 1.0)
        
        H_prop_z_s = torch.tensor(phase_spatial_numpy, dtype=self.dtype_t).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.band_limit:
            theta_y = np.arcsin(self.wavelength / self.LCoS_pitch)
            theta_x = np.arcsin(self.wavelength / self.LCoS_pitch)
            zmaxx = max(x / np.tan(theta_x), abs(prop_z))
            zmaxy = max(y / np.tan(theta_y), abs(prop_z))
            fx_max = 1 / np.sqrt((2 * zmaxx * (1 / x)) ** 2 + 1) / self.wavelength * factor
            fy_max = 1 / np.sqrt((2 * zmaxy * (1 / y)) ** 2 + 1) / self.wavelength * factor
            H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=self.dtype_t).to(self.device)
        else:
            H_filter = torch.tensor(1, dtype=self.dtype_t).to(self.device)
        
        transf_func_in_s = torch.polar(H_filter, H_prop_z_s)
        transf_func_H_s = torch.fft.ifftshift(transf_func_in_s)
        
        return transf_func_H_s, global_phase_term_Hg

    def prop_asm(self, transf_func_H_s, global_phase_term_Hg, u0=None, amp_in=None, phs_in=None):
        # ASM propagation (corrected physical model)
        if u0 is None:
            amp_in = amp_in if amp_in is not None else torch.ones_like(phs_in)
            phs_in = phs_in if phs_in is not None else torch.zeros_like(amp_in)
            u0 = torch.polar(amp_in, phs_in)

        b, c, h, w = u0.size()
        if self.linear_conv:
            u0 = nn.ReflectionPad2d((w // 2, w // 2, h // 2, h // 2))(u0)

        # U1 = torch.fft.fftn(torch.fft.ifftshift(u0) / np.sqrt(h * w), dim=(-2, -1), norm='ortho')
        U1 = torch.fft.fftn(torch.fft.ifftshift(u0), dim=(-2, -1), norm='ortho')
        
        # Core correction: combine spatial and global transfer functions in Fourier domain
        # H_complete = transf_func_H_s * global_phase_term_Hg
        U2 = transf_func_H_s * global_phase_term_Hg * U1
        
        # u_complete = torch.fft.ifftshift(torch.fft.ifftn(U2 * np.sqrt(h * w), dim=(-2, -1), norm='ortho'))
        u_complete = torch.fft.ifftshift(torch.fft.ifftn(U2, dim=(-2, -1), norm='ortho'))
        
        if self.linear_conv:
            u_complete = u_complete[..., int(h / 2):int(h / 2) + h, int(w / 2):int(w / 2) + w]
            
        return u_complete


# if __name__ == '__main__':
#     # Test case
#     optics = Optics(2, 0.010, 0.005*2**0, 2**2)
#
#     # Test distance_int_wavelength_process method
#     test_z = 0.005
#     processed_z = optics.distance_int_wavelength_process(test_z)
#     # Test distance_int_wavelength_process method
#     test_z = 0.005
#     processed_z = optics.distance_int_wavelength_process(test_z)
