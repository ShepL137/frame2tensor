"""Frame-differencing motion detectors shared by the example's entry points.

Both detectors consume an RGBA frame normalized to 0..1 and return an RGBA motion map with a fully opaque alpha channel.
"""
import torch
from torch import nn

# Gains assume input normalized to 0..1.
EMA_ALPHA            = 0.8   # background update rate; higher forgets faster
EMA_K                = 50.0  # tanh gain; ~3 grey levels of change reach mid brightness
ADAPTIVE_ALPHA       = 0.1   # statistics update rate
ADAPTIVE_SENSITIVITY = 3.0   # tanh gain around the visibility threshold
ADAPTIVE_VAR_INIT    = 1e-4  # prior noise floor (std 0.01, ~3 grey levels)


class EMAMotionDetector(nn.Module):
    """Motion as absolute difference against an exponential-moving-average background.

    The background chases the current frame at rate ``alpha``,
    so moving content leaves a bright trail until the background catches up.
    """

    def __init__(self, h, w, alpha=EMA_ALPHA, k=EMA_K):
        super().__init__()
        self.register_buffer("background", torch.zeros(h, w, 3, device="cuda"))
        self.alpha = alpha
        self.k     = k

    def forward(self, frame):
        rgb             = frame[..., :3]
        difference      = torch.abs(rgb - self.background)
        self.background = self.alpha * rgb + (1 - self.alpha) * self.background

        motion          = torch.tanh(difference * self.k)
        alpha_channel   = torch.ones_like(motion[..., :1])

        return torch.cat([motion, alpha_channel], dim=-1)


class AdaptiveMotionDetector(nn.Module):
    """Motion as a per-pixel z-score against running mean and variance.

    Each pixel is compared to its own noise statistics,
    which places the visibility threshold exactly at that pixel's noise floor:
    static regions shimmer while true motion stands out with more intensity.
    """

    def __init__(self, h, w, alpha=ADAPTIVE_ALPHA, sensitivity=ADAPTIVE_SENSITIVITY):
        super().__init__()
        self.register_buffer("mean", torch.zeros(h, w, 3, device="cuda"))
        self.register_buffer("var", torch.full((h, w, 3), ADAPTIVE_VAR_INIT, device="cuda"))
        self.alpha       = alpha
        self.sensitivity = sensitivity

    def forward(self, frame):
        rgb            = frame[..., :3]
        delta          = rgb - self.mean
        self.mean     += self.alpha * delta
        self.var       = (1 - self.alpha) * self.var + self.alpha * delta**2

        std            = torch.sqrt(self.var + 1e-6)
        z_scores       = torch.abs(delta) / std

        motion         = torch.tanh(self.sensitivity * (z_scores - 1.5))
        alpha_channel  = torch.ones_like(motion[..., :1])

        return torch.cat([motion, alpha_channel], dim=-1)


MODELS = {
    "ema"     : EMAMotionDetector,
    "adaptive": AdaptiveMotionDetector,
}
