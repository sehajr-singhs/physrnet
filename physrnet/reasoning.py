"""Graph-based reasoning module.

Builds an entity interaction graph from particle states and applies a graph
network to infer, for every ordered pair of particles, a signed interaction
strength. The predicted acceleration is then the weighted sum of unit
displacement vectors — a differentiable, *interpretable* force model:

    a_i = sum_{j != i}  w_ij · (r_i - r_j) / |r_i - r_j|

The learned edge weights ``w_ij`` are the recovered interaction graph: they
reveal which particles actually exert force on which, the way GNOmE's circuit
graph reveals which units depend on which inputs. Because ``w_ij`` is a
function of rotation-invariant features only and the displacement direction is
rotation-equivariant, this pathway is exactly E(2)-equivariant too.

The whole batch is processed in one fully-vectorized call (no per-sample
Python loops).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EntityGraphBuilder(nn.Module):
    """Constructs a fully-connected entity interaction graph (vectorized).

    Edge features are rotation-invariant (distance, relative speed); the
    displacement direction is returned separately as an equivariant vector.
    """

    def __init__(self, n_particles: int = 3):
        super().__init__()
        self.n = n_particles
        pairs = [(i, j) for i in range(n_particles) for j in range(n_particles) if i != j]
        self.src_local = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.dst_local = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.E = len(pairs)

    def forward(self, pos: torch.Tensor, vel: torch.Tensor):
        """Build the graph for a batch of particle states.

        Args:
            pos: (B, N, D), vel: (B, N, D)
        Returns:
            edge_index: (2, B*E) global (batch-major) src -> dst
            edge_attr: (B*E, 2)  — (distance, relative speed), rotation-invariant
            direction: (B*E, D)  — unit displacement (r_src - r_dst)/|..|, equivariant
        """
        B, N, D = pos.shape
        dev = pos.device

        src_l = self.src_local.to(dev)
        dst_l = self.dst_local.to(dev)

        diff = pos[:, src_l, :] - pos[:, dst_l, :]                    # (B, E, D)
        dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)        # (B, E, 1)
        direction = (diff / dist).reshape(-1, D)                      # (B*E, D)

        rel_vel = vel[:, src_l, :] - vel[:, dst_l, :]                 # (B, E, D)
        edge_attr = torch.cat([dist.squeeze(-1), rel_vel.norm(dim=-1)], dim=-1)  # (B, E, 2)
        edge_attr = edge_attr.reshape(-1, 2)                          # (B*E, 2)

        base = (torch.arange(B, device=dev).unsqueeze(1) * N).expand(B, self.E)
        src = (base + src_l).reshape(-1)
        dst = (base + dst_l).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)                   # (2, B*E)

        return edge_index, edge_attr, direction


class ReasoningGNN(nn.Module):
    """Graph network that recovers the interaction graph and predicts forces.

    Args:
        n_scalar: number of invariant scalar node features.
        hidden: MLP hidden width.
        out_dim: spatial dimensionality of the acceleration (2 for 2D).
    """

    def __init__(self, n_scalar: int = 4, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.out_dim = out_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(n_scalar * 2 + 2, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def forward(self, x_scalar: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, direction: torch.Tensor,
                n_nodes: int):
        """Predict per-node acceleration as a weighted sum of unit displacements.

        Args:
            x_scalar: (N_total, n_scalar) node features (batch-flattened)
            edge_index: (2, E_total)
            edge_attr: (E_total, 2) invariant edge features (distance, rel speed)
            direction: (E_total, D) equivariant unit displacement
            n_nodes: N_total
        Returns:
            accel: (N_total, out_dim) — equivariant
            edge_weights: (E_total, 1) — recovered interaction strengths
        """
        src, dst = edge_index
        if src.numel() == 0:
            return (torch.zeros(n_nodes, self.out_dim, device=x_scalar.device),
                    torch.zeros(0, 1, device=x_scalar.device))

        msg_in = torch.cat([x_scalar[src], x_scalar[dst], edge_attr], dim=-1)
        w = self.edge_mlp(msg_in)                                  # (E_total, 1)

        force = torch.zeros(n_nodes, self.out_dim, device=x_scalar.device)
        force = force.index_add(0, dst, w * direction)             # (N_total, out_dim)

        return force, w
