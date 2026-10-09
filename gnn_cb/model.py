"""Small, permutation-equivariant conditional deformation models.

Only the initial geometry, initial graph and prescribed protocol phase are inputs.
The phase is retrospective video-protocol information, not calibrated strain.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass
class ModelConfig:
    kind: str = "gnn"
    hidden_dim: int = 32
    message_steps: int = 3
    mlp_width: int = 104


def affine_features(x0: torch.Tensor, progress: torch.Tensor) -> torch.Tensor:
    """Return [phase, node, 9] quadratic-phase/affine-space basis."""
    centered = x0 - x0.mean(dim=0, keepdim=True)
    spatial = torch.cat((torch.ones_like(centered[:, :1]), centered), dim=-1)
    temporal = torch.stack((torch.ones_like(progress), progress, progress.square()), -1)
    return (temporal[:, None, :, None] * spatial[None, :, None, :]).flatten(-2)


def dense(inputs: int, hidden: int, outputs: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class MessageBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.message = dense(2 * hidden + 3, hidden, hidden)
        self.update = dense(2 * hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                relative: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        aggregate = torch.zeros_like(h)
        if source.numel():
            edge_attr = relative.unsqueeze(0).expand(h.shape[0], -1, -1)
            messages = self.message(torch.cat((h[:, source], h[:, target], edge_attr), -1))
            aggregate.index_add_(1, target, messages)
            degree = torch.bincount(target, minlength=h.shape[1]).to(h.dtype).clamp_min(1)
            aggregate = aggregate / degree[None, :, None]
        return self.norm(h + self.update(torch.cat((h, aggregate), -1)))


class DeformationModel(nn.Module):
    """Predict dimensionless [dx,dy] at nodes for one graph and B phases.

    ``forward(x0[N,2], edge_index[2,E], progress[B]) -> [B,N,2]``.
    The affine buffer is estimated from the training partition only; the neural
    output is a learned residual. MLP forward deliberately ignores edge_index.
    """
    def __init__(self, config: ModelConfig,
                 affine_coefficients: torch.Tensor | None = None,
                 residual_scale: float = 1.0):
        super().__init__()
        self.config = config
        coefficients = (torch.zeros(9, 2) if affine_coefficients is None
                        else torch.as_tensor(affine_coefficients, dtype=torch.float32).reshape(9, 2))
        self.register_buffer("affine_coefficients", coefficients.clone())
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)))
        if config.kind == "gnn":
            hidden = config.hidden_dim
            self.encoder = dense(4, hidden, hidden)
            self.blocks = nn.ModuleList(MessageBlock(hidden) for _ in range(config.message_steps))
            self.decoder = dense(hidden, hidden, 2)
            last = self.decoder[-1]
        elif config.kind == "mlp":
            width = config.mlp_width
            self.network = nn.Sequential(
                nn.Linear(4, width), nn.SiLU(),
                nn.Linear(width, width), nn.SiLU(),
                nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
            last = self.network[-1]
        else:
            raise ValueError(f"Unknown model kind: {config.kind}")
        # Epoch zero is exactly the training-only affine baseline.
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x0: torch.Tensor, edge_index: torch.Tensor,
                progress: torch.Tensor) -> torch.Tensor:
        progress = progress.reshape(-1)
        centered = x0 - x0.mean(dim=0, keepdim=True)
        xy = centered.unsqueeze(0).expand(progress.numel(), -1, -1)
        phase = progress[:, None, None].expand(-1, x0.shape[0], 1)
        features = torch.cat((xy, phase, phase.square()), -1)
        if self.config.kind == "gnn":
            h = self.encoder(features)
            source, target = edge_index
            delta = x0[source] - x0[target]
            relative = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
            for block in self.blocks:
                h = block(h, edge_index, relative)
            residual = self.decoder(h)
        else:
            residual = self.network(features)
        baseline = affine_features(x0, progress) @ self.affine_coefficients
        return baseline + self.residual_scale * residual

    def metadata(self) -> dict:
        return {"config": asdict(self.config),
                "trainable_parameters": sum(p.numel() for p in self.parameters())}


def load_checkpoint(path: str, device: str = "cpu") -> tuple[DeformationModel, dict]:
    """Load our tensor/dict-only checkpoint safely (no arbitrary pickle classes)."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = DeformationModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint
