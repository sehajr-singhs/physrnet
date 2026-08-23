"""E(3)-equivariant message-passing layers for 3D particle systems.

Implements the EGNN-style (E(n) Equivariant Graph Neural Networks) architecture
with scalar-vector decomposition:

  * Scalar features: rotation-invariant quantities (distances, masses, speeds,
    kinetic energies, dot products).
  * Vector features: rotation-equivariant quantities (displacement vectors,
    velocities, angular momentum contributions).

Message passing is exactly E(3)-equivariant because:
  * All scalar coefficients are functions of rotation-invariant features only
  * Vector messages are scalar_gate * equivariant_vector (preserves equivariance)
  * Node updates use only invariant aggregates of equivariant features

The key improvement over E(2): this operates in full 3D space with proper
rotation group O(3) equivariance, needed for real molecular and physical systems.

References:
  * Satorras, Hoogeboom & Welling, "E(n) Equivariant Graph Neural Networks" (ICML 2021)
  * Brandstetter et al., "Geometric and Physical Quantities improve E(3) MP" (NeurIPS 2022)
  * Thomas et al., "Tensor Field Networks" (NeurIPS 2018)
  * Batatia et al., "MACE: Higher Order Equivariant Message Passing Neural Networks" (NeurIPS 2022)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class EquivariantEdgeNet(nn.Module):
    """Edge network that produces scalar messages and vector gates from
    rotation-invariant edge features."""

    def __init__(self, inv_dim: int, n_scalar: int, n_vec: int, hidden: int):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vec = n_vec
        self.net = nn.Sequential(
            nn.Linear(inv_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_scalar + n_vec),
        )

    def forward(self, inv_features: torch.Tensor):
        """inv_features: (E, inv_dim) -> (E, n_scalar + n_vec)"""
        return self.net(inv_features)


class EquivariantNodeNet(nn.Module):
    """Node update network from invariant aggregates."""

    def __init__(self, in_dim: int, n_scalar: int, n_vec: int, hidden: int):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vec = n_vec
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_scalar + n_vec),
        )

    def forward(self, node_features: torch.Tensor):
        """node_features: (N, in_dim) -> (N, n_scalar + n_vec)"""
        return self.net(node_features)


class E3EquivariantMP(nn.Module):
    """Stack of E(3)-equivariant message-passing layers.

    Each layer:
      1. Compute rotation-invariant edge features (distances, dot products)
      2. Edge MLP -> (scalar_msg, vector_gate)
      3. Vector message = vector_gate * (x_src - x_dst)  [equivariant]
      4. Aggregate messages to destination nodes
      5. Node MLP -> (scalar_delta, vector_update_gate)
      6. Update: s += scalar_delta, x += gate * agg_vector_msg

    Args:
        n_scalar: number of rotation-invariant scalar channels per node.
        hidden: MLP hidden width.
        n_layers: number of message-passing rounds.
        n_vec_channels: number of vector (3D) channels per node.
    """

    def __init__(self, n_scalar: int = 8, hidden: int = 128,
                 n_layers: int = 3, n_vec_channels: int = 2):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vec_channels = n_vec_channels
        self.n_layers = n_layers

        # Invariant edge feature dim:
        #   s_src (n_scalar) + s_dst (n_scalar)
        #   + ||x_src - x_dst|| (1)
        #   + ||v_src - v_dst|| per vec channel (n_vec_channels)
        #   + dot(v_src, v_dst) per vec channel (n_vec_channels)
        inv_dim = n_scalar * 2 + 1 + n_vec_channels * 2

        self.edge_nets = nn.ModuleList()
        self.node_nets = nn.ModuleList()
        for _ in range(n_layers):
            self.edge_nets.append(
                EquivariantEdgeNet(inv_dim, n_scalar, n_vec_channels, hidden))
            # Node net input: s (n_scalar) + ||agg_v|| per channel (n_vec_channels)
            self.node_nets.append(
                EquivariantNodeNet(n_scalar + n_vec_channels, n_scalar,
                                   n_vec_channels, hidden))

    def _edge_features(self, s: torch.Tensor, x: torch.Tensor,
                       v: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        """Compute rotation-invariant edge features.

        Args:
            s: (N, n_scalar) scalar node features
            x: (N, 3) positions (equivariant, used for distance computation)
            v: (N, n_vec_channels * 3) vector node features (flattened)
            src, dst: (E,) edge endpoint indices
        Returns:
            inv: (E, inv_dim) rotation-invariant edge features
        """
        n_ch = self.n_vec_channels

        # Position-based features (the primary equivariant carrier)
        x_src, x_dst = x[src], x[dst]                              # (E, 3)
        diff_x = x_src - x_dst                                     # (E, 3) equivariant
        dist_x = diff_x.norm(dim=-1, keepdim=True)                 # (E, 1) invariant

        # Vector features
        v_src = v[src].view(-1, n_ch, 3)                           # (E, n_ch, 3)
        v_dst = v[dst].view(-1, n_ch, 3)
        diff_v = v_src - v_dst                                     # (E, n_ch, 3) equivariant
        dist_v = diff_v.norm(dim=-1)                               # (E, n_ch) invariant
        dot_v = (v_src * v_dst).sum(-1)                            # (E, n_ch) invariant

        return torch.cat([s[src], s[dst], dist_x, dist_v, dot_v], dim=-1)

    def _mp_step(self, s, x, v, edge_index, n_nodes, edge_net, node_net):
        src, dst = edge_index
        n_ch = self.n_vec_channels

        if src.numel() == 0:
            return s, x, v

        inv = self._edge_features(s, x, v, src, dst)               # (E, inv_dim)
        out = edge_net(inv)                                         # (E, n_scalar + n_ch)
        s_msg = out[:, :self.n_scalar]                              # (E, n_scalar)
        v_gate = out[:, self.n_scalar:]                             # (E, n_ch)

        # Vector message: scalar gate * equivariant displacement
        diff_x = x[src] - x[dst]                                   # (E, 3) equivariant
        v_msg = v_gate.unsqueeze(-1) * diff_x.unsqueeze(1)         # (E, n_ch, 3)
        v_msg = v_msg.reshape(-1, n_ch * 3)                        # (E, n_ch*3)

        # Aggregate to destinations
        agg_s = torch.zeros(n_nodes, self.n_scalar, device=s.device)
        agg_s.index_add_(0, dst, s_msg)
        agg_v = torch.zeros(n_nodes, n_ch * 3, device=v.device)
        agg_v.index_add_(0, dst, v_msg)

        # Node update from invariant aggregates
        agg_v_norm = agg_v.view(-1, n_ch, 3).norm(dim=-1)          # (N, n_ch) invariant
        node_in = torch.cat([s, agg_v_norm], dim=-1)               # (N, n_scalar + n_ch)
        delta = node_net(node_in)                                   # (N, n_scalar + n_ch)

        ds = delta[:, :self.n_scalar]
        dv_gate = torch.sigmoid(delta[:, self.n_scalar:])           # (N, n_ch) invariant gates

        new_s = s + ds
        v_update = (dv_gate.unsqueeze(-1) * agg_v.view(-1, n_ch, 3)).reshape(-1, n_ch * 3)
        new_v = v + v_update
        # x is not updated here — it's the fixed particle positions

        return new_s, x, new_v

    def forward(self, s: torch.Tensor, x: torch.Tensor, v: torch.Tensor,
                edge_index: torch.Tensor, n_nodes: int):
        """Run the E(3)-equivariant stack.

        Args:
            s: (N, n_scalar) rotation-invariant scalar features
            x: (N, 3) positions (equivariant, not updated)
            v: (N, n_vec_channels * 3) vector features (equivariant)
            edge_index: (2, E) src -> dst
            n_nodes: N
        Returns:
            (new_s, x, new_v) — same shapes, exactly E(3)-equivariant.
        """
        for i in range(self.n_layers):
            s, x, v = self._mp_step(s, x, v, edge_index, n_nodes,
                                     self.edge_nets[i], self.node_nets[i])
        return s, x, v
