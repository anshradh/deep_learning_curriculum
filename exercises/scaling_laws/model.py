# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from fancy_einsum import einsum
from dataclasses import dataclass


@dataclass
class MNISTCNNConfig:
    d_kernel: int
    n_filters: int
    n_layers: int
    dropout: float = 0.25


class MNISTCNN(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self, config: MNISTCNNConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=1 if i == 0 else config.n_filters,
                    out_channels=config.n_filters,
                    kernel_size=config.d_kernel,
                    padding="same",
                    bias=False,
                )
                for i in range(config.n_layers)
            ]
        )
        self.fc = nn.Linear(28 * 28 * config.n_filters, 10)

    def forward(self, x):
        for layer in self.layers:
            x = x + F.relu(
                layer(
                    F.dropout(
                        F.layer_norm(
                            x,
                            x.shape[1:] if x.ndim == 4 else x.shape,
                        ),
                        self.config.dropout,
                    ),
                ),
            )

        x = einops.rearrange(x, "... c h w -> ... (c h w)")
        x = self.fc(x)
        return x


# %%
