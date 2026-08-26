"""Relativistic mechanics module for PSN-Universal.

Implements:
  * Special relativistic N-body dynamics
  * Lorentz transformations
  * Relativistic momentum/energy: p = γmv, E = γmc²
  * General relativistic geodesic deviation (simplified)
  
The key difference from Newtonian mechanics:
  * γ = 1/√(1 - v²/c²) becomes significant near c
  * Force = dp/dt = d(γmv)/dt, not = ma
  * Forces must be Lorentz-covariant

References:
  * Einstein, "On the Electrodynamics of Moving Bodies" (1905)
  * Will, "The Confrontation between General Relativity and Experiment" (2014)
  * Greydanus et al., "Hamiltonian Neural Networks" (NeurIPS 2019) - for relativistic extension
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class RelativisticResidual(nn.Module):
    """Special relativistic mechanics residuals.
    
    Core equations:
        γ = 1/√(1 - v²/c²)
        p = γ m v
        E = γ m c²
        dp/dt = F
    """
    
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = c
    
    def gamma(self, vel: torch.Tensor) -> torch.Tensor:
        """Lorentz factor: γ = 1/√(1 - v²/c²).
        
        vel: (..., D) velocity vectors
        Returns: (...) Lorentz factors
        """
        v2 = vel.pow(2).sum(dim=-1)
        beta2 = v2 / (self.c * self.c)
        beta2 = beta2.clamp(max=0.9999)  # prevent NaN
        return 1.0 / torch.sqrt(1.0 - beta2)
    
    def relativistic_momentum(self, vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """p = γmv.
        
        vel: (N, D), mass: (N,)
        Returns: (N, D)
        """
        gamma = self.gamma(vel)
        return gamma.unsqueeze(-1) * mass.unsqueeze(-1) * vel
    
    def relativistic_energy(self, vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """E = γmc²."""
        gamma = self.gamma(vel)
        return gamma * mass * self.c * self.c
    
    def rest_energy(self, mass: torch.Tensor) -> torch.Tensor:
        """E_rest = mc²."""
        return mass * self.c * self.c
    
    def kinetic_energy_rel(self, vel: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """KE = (γ - 1)mc²."""
        gamma = self.gamma(vel)
        return (gamma - 1.0) * mass * self.c * self.c
    
    def momentum_energy_relation(self, p: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        """E² = (pc)² + (mc²)².
        
        Checks the relativistic dispersion relation.
        """
        p2 = p.pow(2).sum(dim=-1)
        m2c4 = (mass * self.c * self.c) ** 2
        E2 = p2 * self.c * self.c + m2c4
        return E2
    
    def lorentz_transform(self, pos: torch.Tensor, vel: torch.Tensor, t: torch.Tensor,
                          boost_vel: torch.Tensor) -> tuple:
        """Apply Lorentz boost to position and time coordinates.
        
        For boost along x-direction with velocity v_b:
            t' = γ(t - v_b x / c²)
            x' = γ(x - v_b t)
        """
        gamma = self.gamma(boost_vel.unsqueeze(0))
        beta = boost_vel / self.c
        
        # Dot product for spatial part
        x_parallel = (pos * beta.unsqueeze(0)).sum(dim=-1, keepdim=True) * beta.unsqueeze(0)
        x_perp = pos - x_parallel
        
        t_prime = gamma * (t.unsqueeze(-1) - (pos * beta.unsqueeze(0)).sum(dim=-1, keepdim=True) / self.c)
        x_prime = gamma * (x_parallel - boost_vel.unsqueeze(0) * t.unsqueeze(-1)) + x_perp
        
        return x_prime.squeeze(-1), t_prime.squeeze(-1)
    
    def physics_loss(self, pos_pred: torch.Tensor, vel_pred: torch.Tensor,
                     pos_curr: torch.Tensor, vel_curr: torch.Tensor,
                     masses: torch.Tensor, force: torch.Tensor,
                     dt: float) -> dict:
        """Relativistic physics residuals.
        
        Checks:
            1. Relativistic equation of motion: dp/dt = F
            2. Energy-momentum relation: E² = (pc)² + (mc²)²
            3. Velocity bound: v < c
            4. Proper time invariance: Δτ² = Δt² - Δx²/c²
        """
        # 1. Momentum conservation
        p_pred = self.relativistic_momentum(vel_pred, masses)
        p_curr = self.relativistic_momentum(vel_curr, masses)
        dp_dt = (p_pred - p_curr) / dt
        
        momentum_loss = (dp_dt - force).pow(2).mean()
        
        # 2. Energy-momentum relation
        E_pred = self.relativistic_energy(vel_pred, masses)
        E_from_p = torch.sqrt(self.momentum_energy_relation(p_pred, masses))
        E_relation_loss = (E_pred - E_from_p).pow(2).mean()
        
        # 3. Velocity bound: penalize v > 0.99c
        v2 = vel_pred.pow(2).sum(dim=-1)
        c2 = self.c * self.c
        speed_violation = torch.relu(v2 - 0.99 * c2).mean()
        
        # 4. Proper time invariance
        dx2 = (pos_pred - pos_curr).pow(2).sum(dim=-1)
        ds2 = (self.c * dt)**2 - dx2
        proper_time_loss = torch.relu(-ds2).mean()  # ds² should be ≥ 0
        
        total = momentum_loss + 0.1 * E_relation_loss + 0.5 * speed_violation + 0.1 * proper_time_loss
        
        return {
            'momentum_loss': momentum_loss,
            'energy_relation_loss': E_relation_loss,
            'speed_violation': speed_violation,
            'proper_time_loss': proper_time_loss,
            'total': total,
        }


def generate_relativistic_nbody(n_particles: int = 10, n_steps: int = 100,
                                 dt: float = 0.01, c: float = 10.0,
                                 seed: int = 42) -> dict:
    """Generate relativistic N-body dynamics.
    
    Uses relativistic gravitational-like interaction.
    c=10 means typical velocities are ~10% of light speed.
    """
    rng = np.random.default_rng(seed)
    
    pos = rng.normal(0, 2, (n_particles, 3)).astype(np.float32)
    vel = rng.normal(0, 0.5, (n_particles, 3)).astype(np.float32)
    masses = rng.uniform(0.5, 2.0, n_particles).astype(np.float32)
    
    traj_pos = [pos.copy()]
    traj_vel = [vel.copy()]
    
    G = 1.0
    softening = 0.5
    
    for _ in range(n_steps - 1):
        # Compute force (Newtonian gravity, but dynamics are relativistic)
        acc = np.zeros_like(pos)
        
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                r_vec = pos[i] - pos[j]
                r = np.sqrt(np.sum(r_vec**2) + softening**2)
                acc[i] += G * masses[j] * r_vec / r**3
        
        # Relativistic velocity update: p_new = p_old + F*dt, then v = p/(γm)
        p = masses[:, None] * vel / np.sqrt(1.0 - np.clip(
            np.sum(vel**2, axis=-1) / (c*c), 0, 0.9999))[:, None]
        
        p_new = p + masses[:, None] * acc * dt
        
        # Recover velocity: v = p/sqrt(m² + p²/c²)
        p2 = np.sum(p_new**2, axis=-1)
        v_new = p_new / np.sqrt(masses[:, None]**2 + p2[:, None] / (c*c))
        
        vel = v_new.astype(np.float32)
        pos = pos + vel * dt
        
        traj_pos.append(pos.copy())
        traj_vel.append(vel.copy())
    
    return {
        'pos': np.stack(traj_pos).astype(np.float32),
        'vel': np.stack(traj_vel).astype(np.float32),
        'masses': masses.astype(np.float32),
        'physics_type': 'relativistic',
        'params': {'c': c, 'G': G, 'softening': softening},
    }


class GRGeodesicResidual(nn.Module):
    """Simplified GR geodesic deviation residual.
    
    The geodesic equation: d²x^μ/dτ² + Γ^μ_νρ (dx^ν/dτ)(dx^ρ/dτ) = 0
    
    For weak-field approximation (Newtonian limit):
        Γ^i_00 ≈ ∂_i Φ = -GM r_i / r³
    """
    
    def __init__(self, G: float = 1.0, c: float = 1.0):
        super().__init__()
        self.G = G
        self.c = c
    
    def christoffel_approx(self, pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Compute approximate Γ^i_00 at each particle position.
        
        Weak-field: Γ^i_00 = ∂_i Φ where Φ = -GM/r
        """
        N = pos.shape[0]
        softening = 0.5
        gamma_00 = torch.zeros(N, 3, device=pos.device)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                r_vec = pos[i] - pos[j]
                r = r_vec.norm()
                r_safe = torch.sqrt(r*r + softening*softening)
                gamma_00[i] += self.G * masses[j] * r_vec / (r_safe**3)
        
        return gamma_00
    
    def geodesic_residual(self, accel_pred: torch.Tensor, vel: torch.Tensor,
                          pos: torch.Tensor, masses: torch.Tensor) -> torch.Tensor:
        """Geodesic equation residual for timelike particles.
        
        a^i + Γ^i_μν v^μ v^ν ≈ 0
        
        In weak field, dominant term: a^i + Γ^i_00 ≈ 0
        """
        gamma_00 = self.christoffel_approx(pos, masses)
        
        # Geodesic equation (leading order)
        res = accel_pred + gamma_00
        return (res ** 2).mean()
    
    def perihelion_precession(self, pos_traj: torch.Tensor,
                               central_mass: torch.Tensor,
                               dt: float) -> torch.Tensor:
        """Estimate perihelion precession from trajectory data.
        
        GR predicts 43 arcseconds/century for Mercury.
        This provides a diagnostic for whether the model captures GR effects.
        """
        # Compute angular momentum and track perihelion advance
        T, N, D = pos_traj.shape
        
        # For each orbit, find the point of closest approach
        r = pos_traj.norm(dim=-1)  # (T, N)
        
        # Find minima of r (perihelia)
        perihelia_times = []
        for n in range(N):
            for t in range(1, T - 1):
                if r[t, n] < r[t-1, n] and r[t, n] < r[t+1, n]:
                    perihelia_times.append(t)
                    break
        
        # In GR, these should precess
        if len(perihelia_times) < 2:
            return torch.tensor(0.0, device=pos_traj.device)
        
        return torch.tensor(0.0, device=pos_traj.device)  # placeholder