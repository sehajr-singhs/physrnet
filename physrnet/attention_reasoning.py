"""Attention-based reasoning module for interpretable interaction discovery.

Uses multi-head attention where:
  * Attention weights are rotation-invariant (computed from scalar features + distances)
  * Force predictions are rotation-equivariant (scalar attention * equivariant displacement)

This preserves exact E(3) equivariance while being more expressive than a
single-MLP edge network.

Key properties:
  * Multi-head attention discovers multiple interaction types
  * Attention patterns directly reveal interaction structure
  * Scalar force magnitudes preserve equivariance
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionReasoningGNN(nn.Module):
    """Multi-head attention reasoning for interpretable force prediction.

    Architecture:
      1. Compute scalar Q, K from rotation-invariant node features
      2. Attention weights = softmax(QK^T/sqrt(d)) — invariant
      3. Compute scalar force magnitudes from invariant edge features
      4. Force = sum_j [attn(i,j) * f_mag(i,j)] * (r_i - r_j)/|r_i - r_j|
         — equivariant because attn and f_mag are invariant, direction is equivariant

    Args:
        n_scalar: per-particle scalar feature dim
        hidden: hidden width
        n_heads: number of attention heads (interaction types)
        out_dim: spatial dimensionality (2 or 3)
    """

    def __init__(self, n_scalar: int = 8, hidden: int = 128,
                 n_heads: int = 4, out_dim: int = 3):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.head_dim = hidden // n_heads

        # Q, K projections (scalar -> hidden, for attention)
        self.W_q = nn.Linear(n_scalar, hidden)
        self.W_k = nn.Linear(n_scalar, hidden)

        # Force magnitude: invariant edge features -> scalar per head
        # Input: s_src, s_dst, distance, relative speed
        force_input_dim = n_scalar * 2 + 2
        self.force_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(force_input_dim, hidden // 2), nn.SiLU(),
                nn.Linear(hidden // 2, 1),
            ) for _ in range(n_heads)
        ])

        # Distance bias (learnable per-head)
        self.dist_bias = nn.Linear(1, n_heads)

    def forward(self, s: torch.Tensor, x: torch.Tensor,
                edge_index: torch.Tensor, n_nodes: int):
        """Predict forces and recover interaction graph.

        Args:
            s: (N, n_scalar) scalar node features
            x: (N, 3) positions
            edge_index: (2, E) src -> dst
            n_nodes: N
        Returns:
            force: (N, out_dim) predicted acceleration (equivariant)
            attn_weights: (E, n_heads) interaction pattern (invariant)
            edge_weights: (E, 1) scalar interaction strengths
        """
        src, dst = edge_index
        E = src.numel()
        if E == 0:
            return (torch.zeros(n_nodes, self.out_dim, device=s.device),
                    torch.zeros(0, self.n_heads, device=s.device),
                    torch.zeros(0, 1, device=s.device))

        hd = self.head_dim

        # Q, K from scalar features (invariant)
        q = self.W_q(s[src]).view(E, self.n_heads, hd)   # (E, n_heads, hd)
        k = self.W_k(s[dst]).view(E, self.n_heads, hd)

        # Attention logits (invariant)
        attn_logits = (q * k).sum(dim=-1) / (hd ** 0.5)  # (E, n_heads)

        # Distance bias
        dist = (x[src] - x[dst]).norm(dim=-1, keepdim=True)  # (E, 1)
        attn_logits = attn_logits + self.dist_bias(dist)      # (E, n_heads)

        # Softmax over incoming edges per destination node
        # Manual scatter-softmax for correctness
        attn_weights = torch.zeros(E, self.n_heads, device=s.device)
        for h in range(self.n_heads):
            logits_h = attn_logits[:, h]
            # For each destination, compute softmax
            max_per_dst = torch.full((n_nodes,), float('-inf'), device=s.device)
            max_per_dst.scatter_reduce_(0, dst, logits_h, reduce='amax')
            exp_vals = torch.exp(logits_h - max_per_dst[dst])
            sum_per_dst = torch.zeros(n_nodes, device=s.device)
            sum_per_dst.scatter_add_(0, dst, exp_vals)
            attn_weights[:, h] = exp_vals / (sum_per_dst[dst] + 1e-8)

        # Force magnitudes from invariant features only
        # Edge features: s_src, s_dst, distance, relative speed
        diff_x = x[src] - x[dst]
        dist_val = diff_x.norm(dim=-1, keepdim=True)  # (E, 1)
        # Relative speed is not directly available, use distance as proxy
        edge_feats = torch.cat([s[src], s[dst], dist_val, dist_val], dim=-1)  # (E, feat)

        force = torch.zeros(n_nodes, self.out_dim, device=s.device)
        edge_weights = torch.zeros(E, 1, device=s.device)

        direction = diff_x / dist_val.clamp(min=1e-6)  # (E, 3) equivariant

        for h in range(self.n_heads):
            # Scalar force magnitude (invariant!)
            f_mag = self.force_mlps[h](edge_feats)          # (E, 1) invariant
            # Force contribution: invariant_scalar * equivariant_direction
            weighted_f = attn_weights[:, h:h+1] * f_mag * direction  # (E, 3)
            force.index_add_(0, dst, weighted_f)
            # Accumulate scalar edge weights
            edge_weights = edge_weights + attn_weights[:, h:h+1] * f_mag.abs()

        return force, attn_weights, edge_weights
