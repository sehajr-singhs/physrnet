"""Physics-informed constraint modules.

Computes residuals of conservation laws — energy, linear momentum, and angular
momentum — from predicted states. These residuals are differentiable, so they
double as auxiliary training losses (the PINN inductive bias) and as
interpretable diagnostics (how much the model drifts off the physical
constraint manifold during rollout).

All three synthetic systems are internally-forced (no external field), so
linear and angular momentum are conserved exactly in the ground-truth
dynamics; total energy is conserved too. The residuals measure how closely a
learnt model reproduces those invariants.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def kinetic_energy(velocities: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
    """KE = 0.5 * sum_i m_i * |v_i|^2.  vel: (B, N, D), masses: (B, N)."""
    return 0.5 * (masses * velocities.pow(2).sum(dim=-1)).sum(dim=-1)  # (B,)


def gravitational_pe(positions: torch.Tensor, masses: torch.Tensor,
                     G: float = 1.0, softening: float = 0.5) -> torch.Tensor:
    """Plummer-softened PE = -G * sum_{i<j} m_i m_j / sqrt(|r|^2 + eps^2)."""
    B, N, D = positions.shape
    pe = torch.zeros(B, device=positions.device)
    for i in range(N):
        for j in range(i + 1, N):
            r = (positions[:, i] - positions[:, j]).norm(dim=-1)
            pe -= G * masses[:, i] * masses[:, j] / torch.sqrt(r.pow(2) + softening ** 2)
    return pe


def spring_pe(positions: torch.Tensor, rest_length: float = 1.0,
              k: float = 1.0) -> torch.Tensor:
    """PE of a linear spring chain: 0.5 k sum_{i} (|r_i - r_{i+1}| - L0)^2."""
    B, N, D = positions.shape
    pe = torch.zeros(B, device=positions.device)
    for i in range(N - 1):
        dist = (positions[:, i] - positions[:, i + 1]).norm(dim=-1)
        pe += 0.5 * k * (dist - rest_length).pow(2)
    return pe


def lennard_jones_pe(positions: torch.Tensor, epsilon: float = 1.0,
                     sigma: float = 1.0) -> torch.Tensor:
    """PE = 4 eps sum_{i<j} [ (sigma/r)^12 - (sigma/r)^6 ].  pos: (B, N, D)."""
    B, N, D = positions.shape
    pe = torch.zeros(B, device=positions.device)
    for i in range(N):
        for j in range(i + 1, N):
            r = (positions[:, i] - positions[:, j]).norm(dim=-1).clamp(min=0.05)
            s6 = (sigma / r).pow(6)
            pe += 4.0 * epsilon * (s6.pow(2) - s6)
    return pe


class PhysicsResidual(nn.Module):
    """Conservation residuals for training and diagnostics.

    Supports gravity (``G``), spring chain (``k``, ``rest_length``), and
    Lennard-Jones (``epsilon``, ``sigma``) potential-energy functions.
    """

    def __init__(self, physics_type: str = "gravity", G: float = 1.0,
                 k: float = 10.0, rest_length: float = 1.0,
                 epsilon: float = 1.0, sigma: float = 1.0,
                 softening: float = 0.5):
        super().__init__()
        self.physics_type = physics_type
        self.G = G
        self.k = k
        self.rest_length = rest_length
        self.epsilon = epsilon
        self.sigma = sigma
        self.softening = softening

    def potential_energy(self, pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        if self.physics_type == "gravity":
            return gravitational_pe(pos, masses, self.G, self.softening)
        if self.physics_type == "spring":
            return spring_pe(pos, self.rest_length, self.k)
        if self.physics_type == "lennard_jones":
            return lennard_jones_pe(pos, self.epsilon, self.sigma)
        raise ValueError(f"unknown physics_type: {self.physics_type}")

    def total_energy(self, pos: torch.Tensor, vel: torch.Tensor,
                     masses: torch.Tensor) -> torch.Tensor:
        return kinetic_energy(vel, masses) + self.potential_energy(pos, masses)

    def momentum(self, vel: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Total linear momentum: sum(m v), shape (B, D)."""
        return (masses.unsqueeze(-1) * vel).sum(dim=1)

    def angular_momentum(self, pos: torch.Tensor, vel: torch.Tensor,
                         masses: torch.Tensor) -> torch.Tensor:
        """Total angular momentum: sum(m r x v), shape (B, 3)."""
        B, N, D = pos.shape
        L = torch.zeros(B, 3, device=pos.device)
        for i in range(N):
            r = pos[:, i]
            v = vel[:, i]
            if D == 2:
                r3 = torch.cat([r, torch.zeros_like(r[:, :1])], dim=-1)
                v3 = torch.cat([v, torch.zeros_like(v[:, :1])], dim=-1)
            else:
                r3, v3 = r, v
            L += masses[:, i].unsqueeze(-1) * torch.cross(r3, v3, dim=-1)
        return L

    def energy_loss(self, pos_pred, vel_pred, pos_true, vel_true, masses):
        """|E_pred - E_true|^2 — total-energy mismatch at the next step."""
        E_pred = self.total_energy(pos_pred, vel_pred, masses)
        E_true = self.total_energy(pos_true, vel_true, masses)
        return (E_pred - E_true).pow(2).mean()

    def momentum_loss(self, vel_pred, vel_true, masses):
        """|p_pred - p_true|^2 — linear-momentum mismatch."""
        p_pred = self.momentum(vel_pred, masses)
        p_true = self.momentum(vel_true, masses)
        return (p_pred - p_true).pow(2).mean()

    def angular_loss(self, pos_pred, vel_pred, pos_true, vel_true, masses):
        """|L_pred - L_true|^2 — angular-momentum mismatch."""
        L_pred = self.angular_momentum(pos_pred, vel_pred, masses)
        L_true = self.angular_momentum(pos_true, vel_true, masses)
        return (L_pred - L_true).pow(2).mean()

    def physics_loss(self, pos_pred, vel_pred, pos_true, vel_true, masses,
                     w_energy: float = 1.0, w_momentum: float = 0.5,
                     w_angular: float = 0.5) -> torch.Tensor:
        """Combined physics-informed loss (energy + linear + angular momentum)."""
        return (w_energy * self.energy_loss(pos_pred, vel_pred, pos_true, vel_true, masses)
                + w_momentum * self.momentum_loss(vel_pred, vel_true, masses)
                + w_angular * self.angular_loss(pos_pred, vel_pred, pos_true, vel_true, masses))
