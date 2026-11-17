"""Generator architecture corresponding to the selected epoch-270 checkpoint."""
from typing import List
import torch
from torch import nn


class TextureResidualBlock(nn.Module):
    def __init__(self, channels: int, residual_scale: float = 0.25) -> None:
        super().__init__()
        self.residual_scale = float(residual_scale)
        groups = max(1, min(8, channels // 8))
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.residual_scale * self.net(x))


class NoiseInjection(nn.Module):
    def __init__(self, channels: int, init_scale: float = 0.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.full((1, channels, 1, 1), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight * noise


class UpsampleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        residual: bool = True,
        dropout: float = 0.0,
        upsample_mode: str = "bilinear",
        norm: str = "instance",
        residual_scale: float = 0.25,
        noise_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if upsample_mode == "nearest":
            upsample: nn.Module = nn.Upsample(scale_factor=2, mode="nearest")
        else:
            upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        layers: List[nn.Module] = [
            upsample,
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=False),
        ]
        if norm == "group":
            layers.append(nn.GroupNorm(max(1, min(8, out_channels // 4)), out_channels))
        elif norm == "instance":
            layers.append(nn.InstanceNorm2d(out_channels, affine=True))
        if noise_scale > 0.0:
            layers.append(NoiseInjection(out_channels, init_scale=noise_scale))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        self.main = nn.Sequential(*layers)
        self.residual = TextureResidualBlock(out_channels, residual_scale=residual_scale) if residual else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(self.main(x))


class ParticleRefinementHead(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, noise_scale: float = 0.035) -> None:
        super().__init__()
        groups = max(1, min(8, hidden_channels // 4))
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            NoiseInjection(hidden_channels, init_scale=noise_scale),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, stride=1, padding=0, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        base_channels: int = 32,
        image_size: int = 512,
        feature_dropout: float = 0.04,
    ) -> None:
        super().__init__()
        if image_size != 512:
            raise ValueError("This WGAN-GP generator is configured for 512x512 output.")
        self.latent_dim = latent_dim
        self.start_size = 16
        c = base_channels
        half_channels = max(c // 2, 8)
        quarter_channels = max(c // 4, 8)
        self.project = nn.Sequential(
            nn.Linear(latent_dim, c * 8 * self.start_size * self.start_size, bias=False),
            nn.BatchNorm1d(c * 8 * self.start_size * self.start_size),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.net = nn.Sequential(
            UpsampleConvBlock(c * 8, c * 4, residual=True, norm="group", residual_scale=0.08, noise_scale=0.0),
            UpsampleConvBlock(c * 4, c * 2, residual=True, norm="group", residual_scale=0.10, noise_scale=0.0),
            UpsampleConvBlock(c * 2, c, residual=True, norm="group", residual_scale=0.12, noise_scale=0.0),
            UpsampleConvBlock(c, half_channels, residual=True, norm="group", residual_scale=0.14, noise_scale=0.006),
            UpsampleConvBlock(half_channels, quarter_channels, residual=True, norm="group", residual_scale=0.15, noise_scale=0.010),
        )
        self.coarse_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(quarter_channels, 1, kernel_size=3, stride=1, padding=0, bias=False),
        )
        self.detail_head = ParticleRefinementHead(
            quarter_channels,
            hidden_channels=max(quarter_channels, 12),
            noise_scale=0.012,
        )
        self.detail_scale = nn.Parameter(torch.tensor(0.12))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.project(z).view(z.size(0), -1, self.start_size, self.start_size)
        features = self.net(x)
        coarse = self.coarse_head(features)
        detail = self.detail_head(features)
        return torch.tanh(coarse + self.detail_scale.clamp(0.0, 0.45) * detail)
