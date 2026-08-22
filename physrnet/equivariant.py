"""E(2)-equivariant message-passing layers for particle systems.

Scalar features (rotation-invariant: mass, speed, distances) and vector
features (rotation-equivariant: centroid-centred position, velocity). Message
passing is exact E(2)-equivariant by construction:

  * scalar messages are MLPs of invariant edge features only,
  * vector messages are ``scalar_gate * (v_src - v_dst)`` — a scalar times an
    equivariant vector, which is again an equivariant vector,
  * node updates multiply aggregated vector messages by invariant scalar gates.

Because every scalar coefficient is a function of rotation-invariant features
only, rotating the input positions/velocities rotates the outputs exactly.

References:
  * Thomas et al., "Tensor Field Networks" (NeurIPS 2018)
  * Satorras, Hoogeboom & Welling, "E(n) Equivariant GNNs" (ICML 2021)
  * Brandstetter et al., "Geometric and Physical Quantities improve E(3) MP" (NeurIPS 2022)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EquivariantMP(nn.Module):
    """A stack of E(2)-equivariant message-passing layers.

    Args:
        n_scalar: number of rotation-invariant scalar channels per node.
        n_vec_channels: number of vector channels (each is a 3-vector, so the
            vector feature dimension is ``n_vec_channels * 3``).
        hidden: MLP hidden width.
        n_layers: number of message-passing rounds (depth).
    """

    def __init__(self, n_scalar: int, n_vec_channels: int, hidden: int = 64,
                 n_layers: int = 2):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_vec_channels = n_vec_channels
        self.n_layers = n_layers

        # Invariant edge-feature dimension:
        #   s_src (n_scalar) + s_dst (n_scalar)
        #   + ||v_src - v_dst|| per channel (n_vec_channels)
        #   + (v_src · v_dst)  per channel (n_vec_channels)
        inv_dim = n_scalar * 2 + n_vec_channels * 2

        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        for _ in range(n_layers):
            # edge network: invariant features -> (scalar msg, per-channel vector gate)
            self.edge_mlps.append(nn.Sequential(
                nn.Linear(inv_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, n_scalar + n_vec_channels)))
            # node network: invariant features -> (scalar delta, per-channel vector gate)
            self.node_mlps.append(nn.Sequential(
                nn.Linear(n_scalar + n_vec_channels, hidden), nn.SiLU(),
                nn.Linear(hidden, n_scalar + n_vec_channels)))

    def _invariant_edge_features(self, s: torch.Tensor, v: torch.Tensor,
                                 edge_index: torch.Tensor) -> torch.Tensor:
        """Build rotation-invariant edge features.

        Args:
            s: (N, n_scalar) invariant scalars
            v: (N, n_vec_channels * 3) vector features (flattened 3-vectors)
            edge_index: (2, E) src -> dst
        """
        src, dst = edge_index
        n_ch = self.n_vec_channels
        v_src = v[src].view(-1, n_ch, 3)
        v_dst = v[dst].view(-1, n_ch, 3)
        diff = v_src - v_dst                                    # (E, n_ch, 3) equivariant
        diff_norm = diff.norm(dim=-1)                           # (E, n_ch) invariant
        dot = (v_src * v_dst).sum(-1)                           # (E, n_ch) invariant
        return torch.cat([s[src], s[dst], diff_norm, dot], dim=-1)  # (E, inv_dim)

    def _mp_step(self, s: torch.Tensor, v: torch.Tensor,
                 edge_index: torch.Tensor, n_nodes: int,
                 edge_mlp: nn.Module, node_mlp: nn.Module):
        src, dst = edge_index
        n_ch = self.n_vec_channels
        n_scalar = self.n_scalar

        if src.numel() == 0:
            return s, v

        inv = self._invariant_edge_features(s, v, edge_index)   # (E, inv_dim)
        out = edge_mlp(inv)                                     # (E, n_scalar + n_ch)
        s_msg = out[:, :n_scalar]                               # (E, n_scalar)
        v_gate = out[:, n_scalar:]                              # (E, n_ch)

        # vector message: scalar gate x (v_src - v_dst), equivariant by construction
        v_src = v[src].view(-1, n_ch, 3)
        v_dst = v[dst].view(-1, n_ch, 3)
        v_msg = v_gate.unsqueeze(-1) * (v_src - v_dst)          # (E, n_ch, 3)
        v_msg = v_msg.reshape(-1, n_ch * 3)

        # aggregate to destinations
        agg_s = torch.zeros(n_nodes, n_scalar, device=s.device).index_add(0, dst, s_msg)
        agg_v = torch.zeros(n_nodes, n_ch * 3, device=v.device).index_add(0, dst, v_msg)

        # node update: invariant input only
        agg_v_norm = agg_v.view(-1, n_ch, 3).norm(dim=-1)       # (N, n_ch) invariant
        node_in = torch.cat([s, agg_v_norm], dim=-1)            # (N, n_scalar + n_ch)
        delta = node_mlp(node_in)                               # (N, n_scalar + n_ch)
        ds = delta[:, :n_scalar]
        dv_gate = torch.sigmoid(delta[:, n_scalar:])            # (N, n_ch) invariant gates

        new_s = s + ds
        v_update = (dv_gate.unsqueeze(-1) * agg_v.view(-1, n_ch, 3)).reshape(-1, n_ch * 3)
        new_v = v + v_update

        return new_s, new_v

    def forward(self, s: torch.Tensor, v: torch.Tensor,
                edge_index: torch.Tensor, n_nodes: int):
        """Run the equivariant stack.

        Args:
            s: (N, n_scalar) scalar features
            v: (N, n_vec_channels * 3) vector features (flattened 3-vectors)
            edge_index: (2, E) src -> dst
            n_nodes: N
        Returns:
            (new_s, new_v) — same shapes, exactly E(2)-equivariant.
        """
        for i in range(self.n_layers):
            s, v = self._mp_step(s, v, edge_index, n_nodes,
                                 self.edge_mlps[i], self.node_mlps[i])
        return s, v
