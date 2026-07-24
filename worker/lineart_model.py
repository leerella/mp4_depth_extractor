from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

CHECKPOINT_URL = "https://huggingface.co/lllyasviel/Annotators/resolve/main/sk_model.pth"
CHECKPOINT_NAME = "lineart_sk_model.pth"
DETECT_RESOLUTION = 512

norm_layer = nn.InstanceNorm2d


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        # attribute name must stay "conv_block" to match the pretrained checkpoint's keys
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), norm_layer(channels), nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1), nn.Conv2d(channels, channels, 3), norm_layer(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class LineartGenerator(nn.Module):
    """Vendored from lllyasviel/ControlNet-v1-1's lineart preprocessor (Apache-2.0, via controlnet_aux).

    Submodule names (model0..model4) must match the pretrained checkpoint's keys.
    """

    def __init__(self, residual_blocks: int = 3) -> None:
        super().__init__()
        self.model0 = nn.Sequential(
            nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), norm_layer(64), nn.ReLU(inplace=True),
        )

        channels = 64
        downsample = []
        for _ in range(2):
            downsample += [nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1), norm_layer(channels * 2), nn.ReLU(inplace=True)]
            channels *= 2
        self.model1 = nn.Sequential(*downsample)

        self.model2 = nn.Sequential(*[_ResidualBlock(channels) for _ in range(residual_blocks)])

        upsample = []
        for _ in range(2):
            upsample += [nn.ConvTranspose2d(channels, channels // 2, 3, stride=2, padding=1, output_padding=1), norm_layer(channels // 2), nn.ReLU(inplace=True)]
            channels //= 2
        self.model3 = nn.Sequential(*upsample)

        self.model4 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(64, 1, 7), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model0(x)
        x = self.model1(x)
        x = self.model2(x)
        x = self.model3(x)
        return self.model4(x)


def load_lineart_model(cache_dir: Path, device: torch.device) -> LineartGenerator:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / CHECKPOINT_NAME
    if not checkpoint.exists():
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
    model = LineartGenerator()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return model.to(device).eval()


def detect_lineart(model: LineartGenerator, frame_bgr: np.ndarray, device: torch.device) -> np.ndarray:
    """Returns a single-channel uint8 frame sized to match frame_bgr: white lines on black."""
    height, width = frame_bgr.shape[:2]
    scale = DETECT_RESOLUTION / min(height, width)
    detect_h = max(64, round(height * scale / 64) * 64)
    detect_w = max(64, round(width * scale / 64) * 64)
    interpolation = cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA
    small = cv2.resize(frame_bgr, (detect_w, detect_h), interpolation=interpolation)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        line = model(tensor)[0, 0].cpu().numpy()
    line = 255 - np.clip(line * 255.0, 0, 255).astype(np.uint8)
    return cv2.resize(line, (width, height), interpolation=cv2.INTER_LINEAR)
