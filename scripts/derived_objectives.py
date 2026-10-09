"""Experimental objectives with observed endpoints only; no new physical labels.

All functions operate on fitting specimens. The predictor still consumes only
initial geometry, topology and phase, including when a global readout is used.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from gnn_cb.model import DeformationModel, affine_features, dense


def temporal_pair_mse(pred, truth, mask, balance_nodes=False):
    """Exact mean squared increment error over all unordered observed time pairs.

    sum_{a<b}(e_a-e_b)^2 = k sum_a(e_a-mean(e))^2 for each node.
    Linear memory/time in the observations; no interpolation across missing data.
    Pair endpoints share the archived node ID, whose physical correctness is not
    independently established. Masked NaNs never enter any subtraction.
    """
    error = torch.where(mask[..., None], pred, 0) - torch.where(mask[..., None], truth, 0)
    count = mask.sum(0)
    pairs = (count * (count - 1) // 2).sum()
    if not pairs:
        return error.sum() * 0
    center = error.sum(0) / count.clamp_min(1)[:, None]
    residual = torch.where(mask[..., None], error - center, 0)
    if balance_nodes:
        # A node with k observations gets mass proportional to k, not k(k-1).
        # Singleton/empty tracks have no valid temporal pair.
        eligible = count > 1
        per_node = residual.square().sum((0, 2)) / (count - 1).clamp_min(1)
        return (per_node[eligible] * count[eligible]).sum() / count[eligible].sum()
    return (count[:, None] * residual.square().sum(0)).sum() / (2 * pairs)


def unique_edges(edge_index):
    # Loader checks symmetric directed edges; use each undirected edge once.
    return edge_index[:, edge_index[0] < edge_index[1]]


def edge_pair_mse(pred, truth, mask, edge_index):
    a, b = unique_edges(edge_index)
    valid = mask[:, a] & mask[:, b]
    if not valid.any():
        return pred.sum() * 0
    # Index before differences: missing targets may contain arbitrary NaNs.
    error = ((pred[:, a][valid] - pred[:, b][valid]) -
             (truth[:, a][valid] - truth[:, b][valid]))
    return error.square().mean()


def objective(pred, truth, mask, edge_index, temporal=0., spatial=0., balance_nodes=False):
    base = (pred[mask] - truth[mask]).square().mean()
    # Each auxiliary mean has a fixed coefficient, regardless of pair count.
    # Division by 2 puts two-endpoint errors on a comparable scale; it does NOT
    # assume independent measurement noise or establish any extra observations.
    if temporal:
        base = base + temporal * temporal_pair_mse(pred, truth, mask, balance_nodes) / 2
    if spatial:
        base = base + spatial * edge_pair_mse(pred, truth, mask, edge_index) / 2
    return base


def jitter_geometry(x0, generator, sigma=.005):
    """Monte Carlo input-error hypothesis, in unchanged initial-height units.

    One view per update, same optimizer update budget. Targets, masks, phases,
    edges and physical identity stay fixed. This is not a new deformed specimen.
    """
    noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    noise = noise.clamp(-3, 3) * sigma
    return x0 + noise - noise.mean(0, keepdim=True)


class GlobalReadoutModel(DeformationModel):
    """Separate all-node-centered local residual and learned graph-wide residual.

    The centering uses predictions on ALL nodes, never observed test targets or
    observation masks. Pooling can communicate globally beyond three edge hops.
    """
    def __init__(self, config, affine_coefficients=None, residual_scale=1.):
        super().__init__(config, affine_coefficients, residual_scale)
        self.global_decoder = dense(config.hidden_dim, config.hidden_dim, 2)
        nn.init.zeros_(self.global_decoder[-1].weight)
        nn.init.zeros_(self.global_decoder[-1].bias)

    def forward(self, x0, edge_index, progress):
        progress = progress.reshape(-1)
        centered = x0 - x0.mean(0, keepdim=True)
        xy = centered.unsqueeze(0).expand(progress.numel(), -1, -1)
        phase = progress[:, None, None].expand(-1, len(x0), 1)
        h = self.encoder(torch.cat((xy, phase, phase.square()), -1))
        a, b = edge_index
        delta = x0[a] - x0[b]
        relative = torch.cat((delta, delta.norm(dim=-1, keepdim=True)), -1)
        for block in self.blocks:
            h = block(h, edge_index, relative)
        local = self.decoder(h)
        residual = local - local.mean(1, keepdim=True) + self.global_decoder(h.mean(1))[:, None]
        return affine_features(x0, progress) @ self.affine_coefficients + self.residual_scale * residual


def label_inventory(data):
    selected = (data['progress'] >= 0) & (data['progress'] <= 1)
    mask = data['mask'][selected]
    count = mask.sum(0)
    edges = data['edge_index']; a, b = edges[:, edges[0] < edges[1]]
    absolute = int(mask.sum())
    temporal = int((count * (count - 1) // 2).sum())
    spatial = int((mask[:, a] & mask[:, b]).sum())
    return {'absolute_vector_targets': absolute,
            'temporal_difference_vector_targets': temporal,
            'spatial_difference_vector_targets': spatial,
            'combined_vector_targets': absolute + temporal + spatial,
            'combined_to_absolute_ratio': (absolute + temporal + spatial) / absolute,
            'temporal_incidence_rank_per_coordinate': int((count[count > 0] - 1).sum()),
            'independent_additional_measurements': 0,
            'warning': 'Derived differences reuse noisy observed endpoints; no masks revived.'}
