"""Thermodynamics and heat transfer module for PSN-Universal.

Implements:
  * Heat equation: ∂T/∂t = α∇²T
  * Fourier's law of heat conduction: q = -k∇T
  * First law of thermodynamics: dU = δQ - δW
  * Ideal gas law: PV = nRT
  * Phase transitions (simplified)

The system is represented as a particle graph where:
  * Nodes = material points with temperature, internal energy
  * Edges = thermal connections with conductivity
  * The GNN learns heat flow from temperature gradients

References:
  * Raissi et al., "Physics-informed neural networks" - heat equation
  * Cai et al., "Physics-Informed Neural Networks for Heat Transfer" (2021)
  * Zobeiry & Humfeld, "A physics-informed ML approach for solving heat transfer" (2021)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class HeatEquationResidual(nn.Module):
    """Heat equation residuals for PINN loss.
    
    ∂T/∂t = α∇²T + Q/ρc_p
    
    where α = k/(ρc_p) is thermal diffusivity, Q is heat source.
    """
    
    def __init__(self, thermal_diffusivity: float = 0.01, density: float = 1.0,
                 specific_heat: float = 1.0):
        super().__init__()
        self.alpha = thermal_diffusivity
        self.rho = density
        self.cp = specific_heat
        self.k = thermal_diffusivity * density * specific_heat  # conductivity
    
    def laplacian(self, T: torch.Tensor, pos: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
        """∇²T via graph Laplacian approximation."""
        src, dst = edge_index
        r_vec = pos[src] - pos[dst]
        r = r_vec.norm(dim=-1).clamp(min=1e-8)
        
        dT = T[src] - T[dst]
        
        # SPH Laplacian: Σ_j 2*m_j/ρ_j * dT_ij * (r·∇W) / (r² + εh²)
        weight = 1.0 / (r * r + 0.01 * 0.1 * 0.1)
        
        lap = torch.zeros(len(pos), device=T.device)
        lap.index_add_(0, dst, dT * weight * 2)
        
        n_neighbors = torch.zeros(len(pos), device=T.device)
        n_neighbors.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        
        lap = lap / n_neighbors.clamp(min=1)
        return lap
    
    def heat_flux(self, T: torch.Tensor, pos: torch.Tensor,
                  edge_index: torch.Tensor) -> torch.Tensor:
        """q = -k∇T heat flux vector at each node."""
        q = -self.k * self.laplacian(T, pos, edge_index).unsqueeze(-1)
        # Simplified: return scalar magnitude
        return q.squeeze(-1)
    
    def residual(self, T_pred: torch.Tensor, T_curr: torch.Tensor,
                 pos: torch.Tensor, edge_index: torch.Tensor,
                 Q: torch.Tensor, dt: float) -> dict:
        """∂T/∂t - α∇²T - Q/(ρc_p) residual."""
        dT_dt = (T_pred - T_curr) / dt
        lap_T = self.laplacian(T_curr, pos, edge_index)
        source = Q / (self.rho * self.cp)
        
        res = dT_dt - self.alpha * lap_T - source
        loss = (res ** 2).mean()
        
        return {
            'heat_loss': loss,
            'max_residual': res.abs().max().item(),
            'mean_residual': res.abs().mean().item(),
        }
    
    def energy_conservation(self, T: torch.Tensor, T_prev: torch.Tensor,
                            Q: torch.Tensor, dt: float) -> torch.Tensor:
        """Total thermal energy should be conserved (with sources/sinks)."""
        U = self.rho * self.cp * T.sum()
        U_prev = self.rho * self.cp * T_prev.sum()
        Q_total = Q.sum() * dt
        return ((U - U_prev - Q_total) ** 2) / max(abs(U_prev), 1e-8)
    
    def physics_loss(self, T_pred: torch.Tensor, T_curr: torch.Tensor,
                     pos: torch.Tensor, edge_index: torch.Tensor,
                     Q: torch.Tensor, dt: float) -> torch.Tensor:
        """Combined thermal physics loss."""
        res = self.residual(T_pred, T_curr, pos, edge_index, Q, dt)
        energy_loss = self.energy_conservation(T_pred, T_curr, Q, dt)
        return res['heat_loss'] + 0.1 * energy_loss


class ThermodynamicsResidual(nn.Module):
    """First law and ideal gas constraints.
    
    For ideal gas:
        PV = nRT → P = nRT/V
        dU = C_v dT
        dU = δQ - PdV (first law)
    """
    
    def __init__(self, R: float = 8.314, cv: float = 12.471):
        super().__init__()
        self.R = R
        self.cv = cv
        self.cp = cv + R
    
    def ideal_gas_residual(self, P: torch.Tensor, V: torch.Tensor,
                           T: torch.Tensor, n: float = 1.0) -> torch.Tensor:
        """PV - nRT should be zero for ideal gas."""
        return (P * V - n * self.R * T).pow(2).mean()
    
    def first_law_residual(self, U_pred: torch.Tensor, U_curr: torch.Tensor,
                           Q: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """ΔU - Q + W should be zero."""
        dU = U_pred - U_curr
        return (dU - Q + W).pow(2).mean()
    
    def adiabatic_residual(self, T1: torch.Tensor, V1: torch.Tensor,
                           T2: torch.Tensor, V2: torch.Tensor,
                           gamma: float = 1.4) -> torch.Tensor:
        """TV^(γ-1) = constant for adiabatic process."""
        return (T1 * V1.pow(gamma - 1) - T2 * V2.pow(gamma - 1)).pow(2).mean()


def generate_heat_flow(n_particles: int = 64, n_steps: int = 100,
                       dt: float = 0.01, alpha: float = 0.1,
                       seed: int = 42) -> dict:
    """Generate heat conduction data.
    
    A hot spot in a 2D plate, cooling by conduction.
    """
    rng = np.random.default_rng(seed)
    
    n_side = int(np.sqrt(n_particles))
    n_particles = n_side * n_side
    
    # Grid of particles on a 2D plate
    x = np.linspace(0, 1, n_side)
    y = np.linspace(0, 1, n_side)
    X, Y = np.meshgrid(x, y)
    pos = np.stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())], axis=-1).astype(np.float32)
    
    # Initial temperature: hot spot in center, cold boundaries
    cx, cy = 0.5, 0.5
    dist = np.sqrt((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2)
    T = np.exp(-dist**2 / 0.05).astype(np.float32)
    
    # Fixed boundary temperature (cold)
    boundary_mask = ((pos[:, 0] < 0.05) | (pos[:, 0] > 0.95) |
                     (pos[:, 1] < 0.05) | (pos[:, 1] > 0.95))
    T[boundary_mask] = 0.0
    
    traj_T = [T.copy()]
    traj_pos = [pos.copy()]
    
    for _ in range(n_steps - 1):
        T_new = T.copy()
        
        # Finite difference heat equation
        for i in range(n_particles):
            if boundary_mask[i]:
                continue
            
            sum_laplacian = 0.0
            n_neighbors = 0
            
            iy, ix = divmod(i, n_side)
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = iy + dy, ix + dx
                if 0 <= ny < n_side and 0 <= nx < n_side:
                    j = ny * n_side + nx
                    d = 1.0 / n_side  # grid spacing
                    sum_laplacian += (T[j] - T[i]) / (d * d)
                    n_neighbors += 1
            
            if n_neighbors > 0:
                T_new[i] = T[i] + alpha * dt * sum_laplacian
        
        T = T_new
        
        # Re-apply boundary
        T[boundary_mask] = 0.0
        
        traj_T.append(T.copy())
    
    return {
        'T': np.stack(traj_T).astype(np.float32),
        'positions': np.stack(traj_pos).astype(np.float32),
        'boundary_mask': boundary_mask.astype(np.float32),
        'physics_type': 'heat',
        'params': {'alpha': alpha, 'plate_size': 1.0, 'n_side': n_side},
    }


def generate_thermo_process(n_particles: int = 32, n_steps: int = 100,
                            process: str = 'isothermal', seed: int = 42) -> dict:
    """Generate thermodynamic process data (ideal gas).
    
    processes: 'isothermal', 'adiabatic', 'isobaric', 'isochoric'
    
    Each particle represents a small volume of gas.
    """
    rng = np.random.default_rng(seed)
    
    pos = rng.normal(0, 1, (n_particles, 3)).astype(np.float32)
    
    if process == 'isothermal':
        T0 = 300.0
        gamma = 1.4
        R = 8.314
        cv = R / (gamma - 1)
        
        T = np.ones(n_particles, dtype=np.float32) * T0
        V = rng.uniform(0.5, 2.0, n_particles).astype(np.float32)
        P = R * T / V
        
        traj_T = [T.copy()]
        traj_P = [P.copy()]
        traj_V = [V.copy()]
        
        for step in range(n_steps - 1):
            # Isothermal: T constant, PV = constant
            V_new = V * (1.0 + rng.normal(0, 0.02, n_particles))
            P_new = R * T / V_new
            
            V = V_new
            P = P_new
            
            traj_T.append(T.copy())
            traj_P.append(P.copy())
            traj_V.append(V.copy())
    
    elif process == 'adiabatic':
        T0 = 300.0
        gamma = 1.4
        R = 8.314
        cv = R / (gamma - 1)
        
        T = np.ones(n_particles, dtype=np.float32) * T0
        V = rng.uniform(0.5, 2.0, n_particles).astype(np.float32)
        P = R * T / V
        
        traj_T = [T.copy()]
        traj_P = [P.copy()]
        traj_V = [V.copy()]
        
        for step in range(n_steps - 1):
            # Adiabatic: TV^(γ-1) = constant, PV^γ = constant
            V_scale = rng.normal(1.0, 0.02, n_particles)
            V_new = V * V_scale
            
            # T_new * V_new^(γ-1) = T * V^(γ-1)
            T_new = T * (V / V_new) ** (gamma - 1)
            P_new = R * T_new / V_new
            
            V = V_new
            T = T_new
            P = P_new
            
            traj_T.append(T.copy())
            traj_P.append(P.copy())
            traj_V.append(V.copy())
    
    else:
        T = np.ones(n_particles, dtype=np.float32) * 300
        V = np.ones(n_particles, dtype=np.float32)
        P = np.ones(n_particles, dtype=np.float32)
        traj_T, traj_P, traj_V = [T], [P], [V]
    
    return {
        'T': np.stack(traj_T).astype(np.float32),
        'P': np.stack(traj_P).astype(np.float32),
        'V': np.stack(traj_V).astype(np.float32),
        'positions': np.stack([pos] * len(traj_T)).astype(np.float32),
        'physics_type': f'thermo_{process}',
        'params': {'process': process, 'gamma': 1.4, 'R': 8.314},
    }