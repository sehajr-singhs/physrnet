"""PhysRNet: a Physics-Aware Reasoning Network.

Three expert modules behind a learned per-node gate:

  * E(2)-equivariant encoder  — exactly rotation-equivariant message passing,
    a "guaranteed-symmetric" function approximator (the E(3)NN pathway).
  * Reasoning GNN            — recovers the *interaction graph* (a signed edge
    weight per particle pair) and predicts force as a weighted sum of unit
    displacements; interpretable, and also exactly equivariant.
  * Physics residual (PINN)  — conservation-law residuals (energy, linear and
    angular momentum) used as auxiliary losses, not as a predictive pathway.

A small meta-network (the gate) blends the two predictive pathways per node:

    a_i = g_i · a_i^equiv  +  (1 - g_i) · a_i^reason

Because the gate's inputs are rotation-invariant scalars and both pathways are
exactly equivariant, the *entire* model is exactly E(2)-equivariant.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .equivariant import EquivariantMP
from .pinn import PhysicsResidual
from .reasoning import EntityGraphBuilder, ReasoningGNN


class PhysRNet(nn.Module):
    """Physics-Aware Reasoning Network for particle systems.

    Args:
        n_particles: number of particles.
        state_dim: per-particle state dim (4 for 2D: x, y, vx, vy).
        physics_type: "gravity", "spring", or "lennard_jones".
        hidden: hidden width for all modules.
        n_scalar: number of rotation-invariant scalar features (4).
    """

    def __init__(self, n_particles: int = 3, state_dim: int = 4,
                 physics_type: str = "gravity", hidden: int = 64,
                 n_scalar: int = 4, freeze_gate: float | None = None):
        super().__init__()
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.pos_dim = state_dim // 2
        self.n_scalar = n_scalar
        self.n_vec_channels = 2  # centroid-centred position, velocity
        # freeze_gate: None = learned blend; 0.0 = reasoning-only; 1.0 = equivariant-only
        self.freeze_gate = freeze_gate

        # Module 1: exactly E(2)-equivariant encoder
        self.equiv_encoder = EquivariantMP(n_scalar, self.n_vec_channels,
                                           hidden, n_layers=2)

        # Module 2: physics residual (auxiliary loss, not a prediction pathway)
        kwargs = {}
        if physics_type == "gravity":
            kwargs = dict(G=1.0, softening=0.5)
        elif physics_type == "spring":
            kwargs = dict(k=10.0, rest_length=1.0)
        elif physics_type == "lennard_jones":
            kwargs = dict(epsilon=1.0, sigma=1.0)
        self.physics = PhysicsResidual(physics_type, **kwargs)

        # Module 3: interpretable reasoning GNN
        self.graph_builder = EntityGraphBuilder(n_particles)
        self.reasoning = ReasoningGNN(n_scalar, hidden, self.pos_dim)

        # Equivariant acceleration head: invariant scalar coefficients x vector channels
        self.vec_head = nn.Sequential(
            nn.Linear(n_scalar, hidden), nn.SiLU(),
            nn.Linear(hidden, self.n_vec_channels))

        # Meta-network: per-node gate between the two pathways
        self.gate = nn.Sequential(
            nn.Linear(n_scalar * 2, hidden), nn.SiLU(),
            nn.Linear(hidden, 1))

    def _build_node_features(self, pos: torch.Tensor, vel: torch.Tensor,
                             masses: torch.Tensor):
        """Build invariant scalar features + equivariant vector features.

        Scalars (rotation-invariant): mass, speed, kinetic energy, centroid distance.
        Vectors (rotation-equivariant, translation-invariant): centroid-centred
        position and velocity.
        """
        B, N, D = pos.shape

        speeds = vel.norm(dim=-1, keepdim=True)                     # (B, N, 1)
        ke = 0.5 * masses.unsqueeze(-1) * vel.pow(2).sum(-1, keepdim=True)
        centroid = pos.mean(dim=1, keepdim=True)                    # (B, 1, D)
        dist_centroid = (pos - centroid).norm(dim=-1, keepdim=True) # (B, N, 1)
        s = torch.cat([masses.unsqueeze(-1), speeds, ke, dist_centroid], dim=-1)

        # centroid-centred position and velocity -> translation-invariant, rotation-equivariant
        v_cm = (masses.unsqueeze(-1) * vel).sum(1, keepdim=True) / masses.sum()
        pos_c = pos - centroid                                       # (B, N, D)
        vel_c = vel - v_cm                                           # (B, N, D)

        def pad3(x):  # (B, N, D) -> (B, N, 3)
            return torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=-1)

        v = torch.cat([pad3(pos_c), pad3(vel_c)], dim=-1)            # (B, N, 6)
        return s, v

    def forward(self, pos: torch.Tensor, vel: torch.Tensor, masses: torch.Tensor):
        """Predict per-particle acceleration.

        Args:
            pos: (B, N, pos_dim), vel: (B, N, pos_dim), masses: (B, N)
        Returns:
            accel_pred: (B, N, pos_dim) — exactly E(2)-equivariant
            edge_weights: (B, E, 1)   — recovered interaction strengths
            gate_vals: (B, N, 1)      — learned per-node pathway blend
        """
        B, N, D = pos.shape
        s, v = self._build_node_features(pos, vel, masses)

        # one fully-connected graph over the whole batch (vectorized)
        edge_index, edge_attr, direction = self.graph_builder(pos, vel)

        # --- equivariant pathway (batched fully-connected graph) ---
        s_flat = s.view(B * N, -1)
        v_flat = v.view(B * N, -1)
        s_enc, v_enc = self.equiv_encoder(s_flat, v_flat, edge_index, B * N)
        s_enc = s_enc.view(B, N, -1)
        v_enc = v_enc.view(B, N, self.n_vec_channels, 3)

        # invariant scalar coefficients x equivariant vector channels -> accel
        coeffs = self.vec_head(s_enc)                               # (B, N, n_vec_channels)
        accel_equiv3 = (coeffs.unsqueeze(-1) * v_enc).sum(dim=2)    # (B, N, 3)
        accel_equiv = accel_equiv3[:, :, :D]                        # (B, N, D)

        # --- reasoning pathway (interpretable interaction graph) ---
        accel_reason_flat, edge_weights = self.reasoning(
            s_flat, edge_index, edge_attr, direction, B * N)
        accel_reason = accel_reason_flat.view(B, N, D)              # (B, N, D)
        edge_weights = edge_weights.view(B, self.graph_builder.E, 1)  # (B, E, 1)

        # --- meta-network gate ---
        if self.freeze_gate is None:
            gate_in = torch.cat([s, s_enc], dim=-1)                 # (B, N, 2*n_scalar)
            g = torch.sigmoid(self.gate(gate_in))                   # (B, N, 1)
        else:
            g = torch.full((B, N, 1), float(self.freeze_gate), device=pos.device)
        accel_pred = g * accel_equiv + (1 - g) * accel_reason       # (B, N, D)

        return accel_pred, edge_weights, g

    def physics_loss(self, pos_pred, vel_pred, pos_true, vel_true, masses):
        return self.physics.physics_loss(pos_pred, vel_pred, pos_true, vel_true, masses)
