"""Spatial retinal color-opponency filter shared by the example's entry point.

Projects an RGBA frame (normalized to 0..1) into three center-surround opponent channels
(luminance, blue-yellow, red-green) in a single fused convolution, then a display remap picks which channel(s) to show.
"""
import torch
import torch.nn.functional as F
from torch import nn

KERNEL_SIZE    = 5
CENTER_SIGMA   = 1.0
SURROUND_SIGMA = 9.0
CONTRAST_GAIN  = 40.0  # sets how hard the opponent responses saturate

RUDERMAN = torch.tensor([
    [0.575,  0.615,  0.540],    # Luminance
    [0.230,  0.250, -0.850],    # Blue-Yellow
    [0.600, -0.630,  0.000],    # Reg-Green
])

# Packed output channel order.
LUM = 0
BY  = 1
RG  = 2

# Display modes, in cycle order.
MODES = ("opponency", "greyscale", "by", "rg")


def _gauss(sigma):
    """Normalized 2D Gaussian on a KERNEL_SIZE grid."""
    ax     = torch.arange(KERNEL_SIZE, device="cuda") - (KERNEL_SIZE - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))

    return kernel / kernel.sum()


class RetinalOpponency(nn.Module):
    """Center-surround color opponency as one fused convolution.

    Each channel is multiplied by the same DoG kernel.
    The module is size-agnostic.

    ``forward`` takes an ``(H, W, 4)`` frame in 0..1 and returns the packed ``(H, W, 3)`` opponent response
        ordered (luminance, blue-yellow, red-green).
    Feed that to :func:`render_mode` to get a displayable RGBA frame.
    """

    def __init__(self):
        super().__init__()
        self.pad = KERNEL_SIZE // 2

        dog    = _gauss(CENTER_SIGMA) - _gauss(SURROUND_SIGMA)      # zero-sum bandpass
        matrix = RUDERMAN.to("cuda")
        weight = matrix[:, :, None, None] * dog[None, None, :, :]   # (out=3, in=3, kH, kW)
        self.register_buffer("weight", weight.contiguous())

    def forward(self, frame):
        rgb      = frame[..., :3].permute(2, 0, 1).unsqueeze(0).contiguous()  # (1, 3, H, W)
        opponent = F.conv2d(rgb, self.weight, padding=self.pad)

        return opponent.squeeze(0).permute(1, 2, 0)  # (H, W, 3)


def _diverge(channel, positive, negative):
    """Map a signed channel to a two-color gradient: positive toward one color, negative the other.

    Zero maps to black.
    """
    device   = channel.device
    positive = channel.clamp(min=0.0) * torch.tensor(positive, device=device)
    negative = (-channel).clamp(min=0.0) * torch.tensor(negative, device=device)

    return positive + negative


def render_mode(opponent, mode, gain=CONTRAST_GAIN):
    """Remap the packed opponent response to a displayable RGBA frame for the given view mode."""
    lum = torch.tanh(gain * opponent[..., LUM:LUM + 1])
    by  = torch.tanh(gain * opponent[..., BY:BY + 1])
    rg  = torch.tanh(gain * opponent[..., RG:RG + 1])

    match mode:
        case "greyscale":
            shade = (lum + 1.0) * 0.5
            rgb   = torch.cat([shade, shade, shade], dim=-1)
        case "by":
            rgb = _diverge(by, positive=(1.0, 1.0, 0.0), negative=(0.0, 0.0, 1.0))
        case "rg":
            rgb = _diverge(rg, positive=(1.0, 0.0, 0.0), negative=(0.0, 1.0, 0.0))
        case _:  # opponency composite: red-green, luminance, blue-yellow into (R, G, B)
            rgb = torch.cat([(rg + 1.0) * 0.5, (lum + 1.0) * 0.5, (by + 1.0) * 0.5], dim=-1)

    alpha = torch.ones_like(rgb[..., :1])

    return torch.cat([rgb, alpha], dim=-1).contiguous()
