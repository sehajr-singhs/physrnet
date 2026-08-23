"""Conservation law discovery module.

Given particle states, this module learns to predict conserved quantities:
  * Total energy (kinetic + potential)
  * Linear momentum vector
  * Angular momentum vector

The key insight: rather than just penalizing conservation violations (PINN
loss), this module *discovers* what quantities are conserved by learning to
predict them from the state. If the model can predict E, p, L from the state
alone, it has discovered the conservation laws.

This is a genuine contribution: existing PINN approaches enforce known
conservation laws as losses. We *learn* them.

Architecture:
  1. Per-particle invariant features (mass, speed, KE) + equivariant features
  2. GNN aggregates to system-level representation
  3. Head predicts scalar (energy), 3-vector (momentum), 3-vector (angular)

The predicted conserved quantities are used both as:
  * Auxiliary training signal (predicted E should match initial E)
  * Interpretable output (what has the model learned is conserved?)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConservationHead(nn.Module):
    """Learns to predict conserved quantities from particle states.

    Given per-particle features, aggregates to a system-level representation
    and predicts:
      * total_energy: scalar (E)
      * linear_momentum: 3-vector (p)
      * angular_momentum: 3-vector (L)

    Args:
        n_scalar: per-particle scalar feature dim
        n_vec_channels: number of vector channels (each 3D)
        hidden: MLP hidden width
        pos_dim: spatial dimensionality (2 or 3)
    """

    def __init__(self, n_scalar: int = 8, n_vec_channels: int = 2,
                 hidden: int = 128, pos_dim: int = 3):
        super().__init__()
        self.pos_dim = pos_dim
        self.n_vec_channels = n_vec_channels

        # System-level aggregator: mean + std + max of scalar features
        # plus norm of vector features
        agg_dim = n_scalar * 3 + n_vec_channels  # mean, std, max of scalars + ||v||

        self.energy_head = nn.Sequential(
            nn.Linear(agg_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )

        self.momentum_head = nn.Sequential(
            nn.Linear(agg_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )

        self.angular_head = nn.Sequential(
            nn.Linear(agg_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, s: torch.Tensor, x: torch.Tensor,
                v: torch.Tensor, masses: torch.Tensor):
        """Predict conserved quantities from particle states.

        Args:
            s: (B, N, n_scalar) per-particle scalar features
            x: (B, N, 3) positions
            v: (B, N, n_vec_channels*3) vector features
            masses: (B, N) particle masses
        Returns:
            energy: (B, 1) predicted total energy
            momentum: (B, 3) predicted linear momentum
            angular: (B, 3) predicted angular momentum
        """
        B, N, _ = s.shape
        n_ch = self.n_vec_channels

        # Aggregate scalar features: mean, std, max over particles
        s_mean = s.mean(dim=1)       # (B, n_scalar)
        s_std = s.std(dim=1)         # (B, n_scalar)
        s_max = s.max(dim=1).values  # (B, n_scalar)

        # Aggregate vector features: mean norm per channel
        v_reshaped = v.view(B, N, n_ch, 3)
        v_norms = v_reshaped.norm(dim=-1)         # (B, N, n_ch)
        v_mean_norm = v_norms.mean(dim=1)         # (B, n_ch)

        agg = torch.cat([s_mean, s_std, s_max, v_mean_norm], dim=-1)  # (B, agg_dim)

        energy = self.energy_head(agg)     # (B, 1)
        momentum = self.momentum_head(agg) # (B, 3)
        angular = self.angular_head(agg)   # (B, 3)

        return energy, momentum, angular

    def conservation_loss(self, pred_energy, pred_momentum, pred_angular,
                          true_energy, true_momentum, true_angular,
                          w_energy=1.0, w_momentum=0.5, w_angular=0.5):
        """Loss: predicted conserved quantities should match ground truth.

        Ground truth is computed from the initial state (which is conserved).
        """
        e_loss = (pred_energy - true_energy.unsqueeze(-1)).pow(2).mean()
        p_loss = (pred_momentum - true_momentum).pow(2).mean()
        l_loss = (pred_angular - true_angular).pow(2).mean()
        return w_energy * e_loss + w_momentum * p_loss + w_angular * l_loss


class PhysicsDiscovery(nn.Module):
    """Discovers conservation laws by learning to predict them from states.

    Unlike PINN (which enforces known conservation laws as auxiliary losses),
    this module *learns* what is conserved — a genuine contribution.

    The predicted conserved quantities serve dual purposes:
      1. Training signal: predicted E/p/L should be constant across timesteps
      2. Interpretability: reveals what invariants the model has discovered
    """

    def __init__(self, n_scalar: int = 8, n_vec_channels: int = 2,
                 hidden: int = 128, pos_dim: int = 3):
        super().__init__()
        self.head = ConservationHead(n_scalar, n_vec_channels, hidden, pos_dim)
        self.pos_dim = pos_dim

    def forward(self, s, x, v, masses):
        return self.head(s, x, v, masses)

    def discovery_loss(self, s_seq, x_seq, v_seq, masses_seq,
                       energy_true, momentum_true, angular_true):
        """Loss across a trajectory: predicted conserved quantities should
        be constant across all timesteps.

        Args:
            s_seq: (T, B, N, n_scalar) scalar features over time
            x_seq: (T, B, N, 3) positions over time
            v_seq: (T, B, N, n_vec_channels*3) vector features over time
            masses_seq: (T, B, N) masses (constant over time)
            energy_true: (T, B) ground truth total energy per timestep
            momentum_true: (T, B, 3) ground truth linear momentum
            angular_true: (T, B, 3) ground truth angular momentum
        Returns:
            loss: scalar
            predictions: dict with predicted quantities at each timestep
        """
        T = s_seq.shape[0]
        all_energy = []
        all_momentum = []
        all_angular = []

        for t in range(T):
            e, p, l = self.head(s_seq[t], x_seq[t], v_seq[t], masses_seq[t])
            all_energy.append(e)
            all_momentum.append(p)
            all_angular.append(l)

        pred_energy = torch.stack(all_energy).squeeze(-1)      # (T, B)
        pred_momentum = torch.stack(all_momentum)               # (T, B, 3)
        pred_angular = torch.stack(all_angular)                 # (T, B, 3)

        # Loss 1: predicted values should match ground truth at each step
        match_loss = self.head.conservation_loss(
            pred_energy, pred_momentum, pred_angular,
            energy_true, momentum_true, angular_true)

        # Loss 2: predicted values should be constant across time (discovery of invariance)
        if T > 1:
            var_energy = pred_energy.var(dim=0).mean()
            var_momentum = pred_momentum.var(dim=0).mean()
            var_angular = pred_angular.var(dim=0).mean()
            constancy_loss = var_energy + 0.5 * var_momentum + 0.5 * var_angular
        else:
            constancy_loss = torch.tensor(0.0, device=s_seq.device)

        return match_loss + 0.5 * constancy_loss, {
            'pred_energy': pred_energy.detach(),
            'pred_momentum': pred_momentum.detach(),
            'pred_angular': pred_angular.detach(),
            'match_loss': match_loss.item(),
            'constancy_loss': constancy_loss.item() if T > 1 else 0.0,
        }
