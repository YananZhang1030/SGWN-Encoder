"""Neural network modules used by SGWN-Encoder."""

import torch
import torch.nn.functional as F
from torch import nn


class ResConvBlock(nn.Module):
    """Two-layer residual convolution block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.final_relu = nn.LeakyReLU(0.2, True)
        self._initialize_weights()

    def forward(self, x):
        return self.final_relu(self.conv_block(x) + self.shortcut(x))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class TransposeConv(nn.Module):
    """2x transpose-convolution upsampling block."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, x):
        return self.transpose_conv(x)


class TPN_R(nn.Module):
    """Compact phase encoder used by the released SGWN-Encoder checkpoint."""

    def __init__(self, in_ch=24, out_ch=4):
        super().__init__()
        n1 = 4
        filters = [n1, n1 * 3, n1 * 6]

        # Attribute names are kept for compatibility with released state_dict files.
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv0_0 = ResConvBlock(in_ch, filters[0])
        self.conv1_0 = ResConvBlock(filters[1], filters[0])
        self.transposeConv1_0 = TransposeConv(filters[0], filters[0])
        self.conv9_9 = nn.Conv2d(filters[2], out_ch, kernel_size=1, stride=1, padding=0)
        self.shuffle = nn.PixelShuffle(2)
        self.unShuffle = nn.PixelUnshuffle(2)

    def forward(self, input):
        input_info = torch.cat((torch.real(input), torch.imag(input)), dim=-3)
        input_unshuffle = self.unShuffle(input_info)

        o0_0 = self.conv0_0(
            torch.cat((torch.cos(input_unshuffle), torch.sin(input_unshuffle), input_unshuffle), dim=-3)
        )
        i1_0 = self.maxPool(o0_0)
        o1_0 = self.conv1_0(torch.cat((torch.cos(i1_0), torch.sin(i1_0), i1_0), dim=-3))
        i9_9 = torch.cat((o0_0, self.transposeConv1_0(o1_0)), dim=1)
        output = self.conv9_9(torch.cat((torch.cos(i9_9), torch.sin(i9_9), i9_9), dim=-3))
        return self.shuffle(output)


class EncodeNet(nn.Module):
    """Experimental complex-input phase encoder."""

    def __init__(self):
        super().__init__()
        base_ch = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(8, base_ch, 3, 1, 1)
        self.body = nn.Sequential(
            ResConvBlock(base_ch, base_ch * 2),
            ResConvBlock(base_ch * 2, base_ch * 2),
        )
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1, 1, 0)
        self.tail = nn.Conv2d(base_ch, 8, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x_in = torch.cat((x.real, x.imag), dim=1)
        x_un = self.unshuffle(x_in)
        f1 = self.head(x_un)
        f2 = self.body(f1)
        out_features = self.fusion_conv(f2) + f1
        phase_vec = self.shuffle(self.tail(out_features))
        return torch.atan2(phase_vec[:, 0:1], phase_vec[:, 1:2])


class EncodeNet_V2(nn.Module):
    """Experimental smooth phase encoder with bounded output range."""

    def __init__(self):
        super().__init__()
        base_ch = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(8, base_ch, 3, 1, 1)
        self.body = nn.Sequential(
            ResConvBlock(base_ch, base_ch * 2),
            ResConvBlock(base_ch * 2, base_ch * 2),
        )
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1, 1, 0)
        self.tail = nn.Conv2d(base_ch, 4, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.normal_(self.tail.weight, std=0.01)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        x_in = torch.cat((x.real, x.imag), dim=1)
        x_un = self.unshuffle(x_in)
        f1 = self.head(x_un)
        f2 = self.body(f1)
        out_features = self.fusion_conv(f2) + f1
        smooth_phase = torch.tanh(self.shuffle(self.tail(out_features))) * 2.0
        return smooth_phase - smooth_phase.mean(dim=(-2, -1), keepdim=True)


class EncodeNet_V3(nn.Module):
    """Experimental unit-vector phase encoder."""

    def __init__(self):
        super().__init__()
        base_ch = 32
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(8, base_ch, 3, 1, 1)
        self.body = nn.Sequential(
            ResConvBlock(base_ch, base_ch * 2),
            ResConvBlock(base_ch * 2, base_ch * 2),
            ResConvBlock(base_ch * 2, base_ch * 2),
        )
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1, 1, 0)
        self.refine = ResConvBlock(base_ch, base_ch)
        self.tail = nn.Conv2d(base_ch, 8, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.normal_(self.tail.weight, std=0.01)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        x_in = torch.cat((x.real, x.imag), dim=1)
        x_un = self.unshuffle(x_in)
        f1 = self.head(x_un)
        f2 = self.body(f1)
        out_features = self.refine(self.fusion_conv(f2) + f1)
        phase_vec = self.shuffle(self.tail(out_features))

        y = 2.0 * torch.tanh(phase_vec[:, 0:1])
        x_component = 1.0 + 2.0 * torch.tanh(phase_vec[:, 1:2])
        unit_vec = F.normalize(torch.cat((y, x_component), dim=1), dim=1, eps=1e-6)
        phase = torch.atan2(unit_vec[:, 0:1], unit_vec[:, 1:2])
        return phase - phase.mean(dim=(-2, -1), keepdim=True)


class EncodeNet_V3_Small(nn.Module):
    """Slimmer V3 variant with the same phase output semantics."""

    def __init__(self):
        super().__init__()
        base_ch = 24
        self.unshuffle = nn.PixelUnshuffle(2)
        self.head = nn.Conv2d(8, base_ch, 3, 1, 1)
        self.body = nn.Sequential(
            ResConvBlock(base_ch, base_ch * 2),
            ResConvBlock(base_ch * 2, base_ch * 2),
        )
        self.fusion_conv = nn.Conv2d(base_ch * 2, base_ch, 1, 1, 0)
        self.tail = nn.Conv2d(base_ch, 8, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        nn.init.normal_(self.tail.weight, std=0.01)
        nn.init.constant_(self.tail.bias, 0)

    def forward(self, x):
        x_in = torch.cat((x.real, x.imag), dim=1)
        x_un = self.unshuffle(x_in)
        f1 = self.head(x_un)
        f2 = self.body(f1)
        phase_vec = self.shuffle(self.tail(self.fusion_conv(f2) + f1))

        y = 2.0 * torch.tanh(phase_vec[:, 0:1])
        x_component = 1.0 + 2.0 * torch.tanh(phase_vec[:, 1:2])
        unit_vec = F.normalize(torch.cat((y, x_component), dim=1), dim=1, eps=1e-6)
        phase = torch.atan2(unit_vec[:, 0:1], unit_vec[:, 1:2])
        return phase - phase.mean(dim=(-2, -1), keepdim=True)
