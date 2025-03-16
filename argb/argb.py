"""Implement pixel-wise encoder from "Rethinking RGB Color Representation for Image Restoration Models" (https://arxiv.org/abs/2402.03399v1)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

torch._dynamo.config.capture_dynamic_output_shape_ops = True


class ArgbPixelEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 32,
        expansion_rate: int = 2,
        num_experts: int = 16,
    ):
        super(ArgbPixelEncoder, self).__init__()
        self.num_experts: int = num_experts
        self.encoder_channels = channels * (expansion_rate**2)

        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels * num_experts,
                channels * num_experts,
                kernel_size=3,
                padding=1,
                groups=num_experts,
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels * num_experts,
                channels * expansion_rate * num_experts,
                kernel_size=3,
                padding=1,
                groups=num_experts,
            ),
            nn.ReLU(),
            nn.Conv2d(
                channels * expansion_rate * num_experts,
                self.encoder_channels * num_experts,
                kernel_size=3,
                padding=1,
                groups=num_experts,
            ),
        )
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels * expansion_rate, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels * expansion_rate),
            nn.ReLU(),
            nn.Conv2d(
                channels * expansion_rate,
                channels * (expansion_rate**2),
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(channels * (expansion_rate**2)),
            nn.ReLU(),
            nn.Conv2d(
                channels * (expansion_rate**2),
                channels * (expansion_rate**3),
                kernel_size=1,
            ),
            nn.BatchNorm2d(channels * (expansion_rate**3)),
            nn.ReLU(),
            nn.Conv2d(channels * (expansion_rate**3), num_experts, kernel_size=1),
            nn.BatchNorm2d(num_experts),
            nn.Sigmoid(),
        )
        self.decoder = nn.Conv2d(
            channels * (expansion_rate**2), in_channels, kernel_size=1
        )
        self.register_buffer("balances", torch.zeros(num_experts))

    @torch.no_grad()
    def update_balance(self, top_experts: torch.Tensor) -> torch.Tensor:
        """Loss-Free Expert Balancing from DeepSeekV3."""
        counts = torch.bincount(
            top_experts.flatten(), minlength=self.num_experts
        ).float()
        mean_count = counts.mean()
        self.balances = self.balances + 0.001 * torch.sign(mean_count - counts)

    @torch.compile
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # route input to experts
        logits = self.router(x) + self.balances.reshape(1, -1, 1, 1)
        _, top_expert = torch.topk(logits, k=1, dim=1)
        if self.training:
            self.update_balance(top_expert)
        # encode input through grouped convolution, input channels need to be repeated for each expert
        encoded_grouped = self.encoder(x.repeat(1, self.num_experts, 1, 1)).reshape(
            x.shape[0], self.num_experts, self.encoder_channels, x.shape[2], x.shape[3]
        )

        # create masks for each expert
        masks = torch.arange(self.num_experts, device=x.device).reshape(
            1, -1, 1, 1, 1
        ).repeat(1, 1, 1, x.shape[2], x.shape[3]) == top_expert.unsqueeze(2)

        # sum over experts
        encoded = (masks * encoded_grouped).sum(dim=1)
        return encoded

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode input
        encoded = self.encode(x)

        # decode by adding a bit of noise
        decoded = self.decoder(encoded + 0.1 * torch.randn_like(encoded))
        loss = F.mse_loss(decoded, x)
        return decoded, loss

    @torch.compile
    def loss(self, x: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # encode inputs
        x = self.encode(x)
        image = self.encode(image)
        return F.mse_loss(x, image)

    @staticmethod
    def from_pretrained(path: str):
        model = ArgbPixelEncoder()
        model.load_state_dict(torch.load(path))
        return model
