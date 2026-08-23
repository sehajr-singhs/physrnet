"""PSN-1 v2: Physics Systems Network, NMI-quality architecture.

Combines three modules behind a learned per-node gate:

  1. E(3)-equivariant encoder — exactly rotation-equivariant message passing
     in full 3D (EGNN-style scalar-vector tensor product).
  2. Attention reasoning GNN — multi-head attention discovers interaction
     types and predicts forces as attention-weighted displacements.
  3. Conservation law discovery — learns to predict conserved quantities
     (energy, momentum, angular momentum) from particle states.

  Plus:
  4. PINN loss — physics-informed residuals for energy/momentum conservation.

The gate blends the two predictive pathways:

    a_i = g_i * a_i^equiv  +  (1 - g_i) * a_i^attn

Because both pathways are exactly E(3)-equivariant and the gate's inputs are
rotation-invariant, the entire model is exactly E(3)-equivariant.

This is the NMI-quality version with:
  * True 3D equivariance (not just E(2) with padding)
  * Attention-based reasoning (more expressive than single-MLP)
  * Conservation law discovery (genuine contribution)
  * Comprehensive ablation support
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .e3_equivariant import E3EquivariantMP
from .attention_reasoning import AttentionReasoningGNN
from .conservation import PhysicsDiscovery
from .pinn import PhysicsResidual


class PSN1v2(nn.Module):
    """Physics Systems Network v2: E(3) + Attention + Conservation Discovery.

    Args:
        n_particles: number of particles.
        state_dim: per-particle state dim (6 for 3D: x, y, z, vx, vy, vz).
        physics_type: "gravity", "spring", or "lennard_jones".
        hidden: hidden width for all modules.
        n_scalar: number of rotation-invariant scalar features.
        n_heads: number of attention heads in reasoning module.
        freeze_gate: None = learned, 0.0 = reasoning-only, 1.0 = equivariant-only.
        use_conservation: whether to use conservation discovery loss.
    """

    def __init__(self, n_particles: int = 5, state_dim: int = 6,
                 physics_type: str = "gravity", hidden: int = 128,
                 n_scalar: int = 8, n_heads: int = 4,
                 freeze_gate: float | None = None,
                 use_conservation: bool = True):
        super().__init__()
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.pos_dim = state_dim // 2
        self.n_scalar = n_scalar
        self.n_heads = n_heads
        self.freeze_gate = freeze_gate
        self.use_conservation = use_conservation

        # Module 1: E(3)-equivariant encoder
        self.equiv_encoder = E3EquivariantMP(
            n_scalar=n_scalar, hidden=hidden, n_layers=3, n_vec_channels=2)

        # Module 2: Attention reasoning GNN
        self.attention_reasoning = AttentionReasoningGNN(
            n_scalar=n_scalar, hidden=hidden, n_heads=n_heads,
            out_dim=self.pos_dim)

        # Module 3: Conservation law discovery
        if use_conservation:
            self.conservation = PhysicsDiscovery(
                n_scalar=n_scalar, n_vec_channels=2, hidden=hidden,
                pos_dim=self.pos_dim)

        # Module 4: PINN (auxiliary loss)
        kwargs = {}
        if physics_type == "gravity":
            kwargs = dict(G=1.0, softening=0.5)
        elif physics_type == "spring":
            kwargs = dict(k=10.0, rest_length=1.0)
        elif physics_type == "lennard_jones":
            kwargs = dict(epsilon=1.0, sigma=1.0)
        self.pinn = PhysicsResidual(physics_type, **kwargs)

        # Equivariant acceleration head
        self.vec_head = nn.Sequential(
            nn.Linear(n_scalar, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),  # 2 vector channels
        )

        # Meta-network gate
        self.gate = nn.Sequential(
            nn.Linear(n_scalar * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def _build_node_features(self, pos, vel, masses):
        """Build invariant scalar features + equivariant vector features.

        Scalars (rotation-invariant): mass, speed, KE, centroid distance,
        pairwise distance stats, potential energy estimate.
        Vectors (rotation-equivariant): centroid-centred position and velocity.
        """
        B, N, D = pos.shape

        speeds = vel.norm(dim=-1, keepdim=True)
        ke = 0.5 * masses.unsqueeze(-1) * vel.pow(2).sum(-1, keepdim=True)
        centroid = pos.mean(dim=1, keepdim=True)
        dist_centroid = (pos - centroid).norm(dim=-1, keepdim=True)

        # Pairwise distance features
        if N > 1:
            pdist = torch.cdist(pos, pos)                           # (B, N, N)
            # Mask self-distances with large finite value
            eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
            pdist_masked = pdist.masked_fill(eye, 1e6)
            mean_dist = pdist_masked.mean(dim=-1, keepdim=True)     # (B, N, 1)
            min_dist = pdist_masked.min(dim=-1).values.unsqueeze(-1)  # (B, N, 1)
            inv_dist = (1.0 / pdist_masked.clamp(min=0.1)).mean(dim=-1, keepdim=True)
        else:
            mean_dist = torch.zeros(B, N, 1, device=pos.device)
            min_dist = torch.ones(B, N, 1, device=pos.device)
            inv_dist = torch.zeros(B, N, 1, device=pos.device)

        s = torch.cat([masses.unsqueeze(-1), speeds, ke, dist_centroid,
                        mean_dist, min_dist, inv_dist], dim=-1)
        # Pad or truncate to n_scalar
        if s.shape[-1] < self.n_scalar:
            s = torch.cat([s, torch.zeros(B, N, self.n_scalar - s.shape[-1],
                                           device=pos.device)], dim=-1)
        else:
            s = s[:, :, :self.n_scalar]
        # Normalize for numerical stability (log-scale for large values)
        s = torch.log1p(s.clamp(min=0))

        # Centroid-centred position and velocity (translation-invariant, rotation-equivariant)
        v_cm = (masses.unsqueeze(-1) * vel).sum(1, keepdim=True) / masses.sum()
        pos_c = pos - centroid
        vel_c = vel - v_cm

        def pad3(x):
            if x.shape[-1] == 3:
                return x
            return torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=-1)

        v = torch.cat([pad3(pos_c), pad3(vel_c)], dim=-1)  # (B, N, 6)
        return s, v

    def _build_graph(self, pos, B, N):
        """Build fully-connected graph edges."""
        dev = pos.device
        pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
        if not pairs:
            return (torch.zeros(2, 0, dtype=torch.long, device=dev), 0)

        src_local = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
        dst_local = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)

        base = (torch.arange(B, device=dev).unsqueeze(1) * N).expand(B, len(pairs))
        src = (base + src_local).reshape(-1)
        dst = (base + dst_local).reshape(-1)
        edge_index = torch.stack([src, dst], dim=0)
        return edge_index, B * N

    def forward(self, pos, vel, masses):
        """Predict per-particle acceleration.

        Args:
            pos: (B, N, 3) positions
            vel: (B, N, 3) velocities
            masses: (B, N) particle masses
        Returns:
            accel_pred: (B, N, 3) predicted acceleration
            attn_weights: interaction pattern from attention reasoning
            edge_weights: scalar interaction strengths
            gate_vals: per-node gate values
        """
        B, N, D = pos.shape
        s, v = self._build_node_features(pos, vel, masses)
        edge_index, n_nodes = self._build_graph(pos, B, N)

        # --- E(3)-equivariant pathway ---
        s_flat = s.view(B * N, -1)
        v_flat = v.view(B * N, -1)
        x_flat = pos.reshape(B * N, 3)

        s_enc, _, v_enc = self.equiv_encoder(s_flat, x_flat, v_flat,
                                              edge_index, B * N)
        s_enc = s_enc.view(B, N, -1)
        v_enc = v_enc.view(B, N, 2, 3)

        coeffs = self.vec_head(s_enc)                              # (B, N, 2)
        accel_equiv3 = (coeffs.unsqueeze(-1) * v_enc).sum(dim=2)   # (B, N, 3)

        # --- Attention reasoning pathway ---
        accel_attn, attn_weights, edge_weights = self.attention_reasoning(
            s_flat, x_flat, edge_index, B * N)
        accel_attn = accel_attn.view(B, N, D)

        # --- Gate ---
        if self.freeze_gate is None:
            gate_in = torch.cat([s, s_enc], dim=-1)
            g = torch.sigmoid(self.gate(gate_in))
        else:
            g = torch.full((B, N, 1), float(self.freeze_gate), device=pos.device)

        accel_pred = g * accel_equiv3 + (1 - g) * accel_attn

        return accel_pred, attn_weights, edge_weights, g

    def pinn_loss(self, pos_pred, vel_pred, pos_true, vel_true, masses):
        """PINN auxiliary loss: conservation law residuals."""
        return self.pinn.physics_loss(pos_pred, vel_pred, pos_true, vel_true, masses)

    def conservation_loss(self, s_seq, x_seq, v_seq, masses_seq,
                          energy_true, momentum_true, angular_true):
        """Conservation law discovery loss (if enabled)."""
        if not self.use_conservation:
            return torch.tensor(0.0, device=s_seq.device), {}
        return self.conservation.discovery_loss(
            s_seq, x_seq, v_seq, masses_seq,
            energy_true, momentum_true, angular_true)
