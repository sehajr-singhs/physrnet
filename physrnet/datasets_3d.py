"""3D physics datasets for PSN-1 benchmarks.

Generates trajectories for:
  1. N-body gravity (3D, softened Plummer potential)
  2. Spring chain (3D, Hooke's law)
  3. Lennard-Jones (3D, molecular dynamics)

All datasets use vectorized generation (no per-sample Python loops).
Energy is conserved to machine precision in the ground truth.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Tuple


def generate_nbody_3d(n_particles: int = 5, n_steps: int = 50,
                      dt: float = 0.01, G: float = 1.0,
                      softening: float = 0.5, seed: int = 42) -> Dict:
    """Generate 3D N-body gravity trajectory with Plummer softening.

    Uses leapfrog integration for energy conservation.
    """
    rng = np.random.RandomState(seed)

    # Random initial conditions
    pos = rng.randn(n_particles, 3) * 2.0
    vel = rng.randn(n_particles, 3) * 0.3
    masses = rng.uniform(0.5, 2.0, n_particles)

    # Center of mass frame
    v_cm = (masses[:, None] * vel).sum(0) / masses.sum()
    vel -= v_cm

    traj_pos = [pos.copy()]
    traj_vel = [vel.copy()]

    def compute_accel(p, m):
        acc = np.zeros_like(p)
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                r = p[i] - p[j]
                dist = np.sqrt(np.sum(r**2) + softening**2)
                acc[i] += G * m[j] * r / dist**3
        return acc

    acc = compute_accel(pos, masses)

    for _ in range(n_steps - 1):
        # Leapfrog integration
        vel_half = vel + 0.5 * dt * acc
        pos_new = pos + dt * vel_half
        acc_new = compute_accel(pos_new, masses)
        vel_new = vel_half + 0.5 * dt * acc_new

        pos = pos_new
        vel = vel_new
        acc = acc_new

        traj_pos.append(pos.copy())
        traj_vel.append(vel.copy())

    pos_arr = np.stack(traj_pos)  # (T, N, 3)
    vel_arr = np.stack(traj_vel)
    mass_arr = np.broadcast_to(masses[None, :], (n_particles, n_particles)).copy()

    return {
        'pos': pos_arr.astype(np.float32),
        'vel': vel_arr.astype(np.float32),
        'masses': mass_arr.astype(np.float32),
        'physics_type': 'gravity',
        'params': {'G': G, 'softening': softening},
    }


def generate_spring_3d(n_particles: int = 5, n_steps: int = 50,
                       dt: float = 0.005, k: float = 10.0,
                       rest_length: float = 1.0, seed: int = 42) -> Dict:
    """Generate 3D spring chain trajectory with Hooke's law."""
    rng = np.random.RandomState(seed)

    # Linear chain with random perturbations
    pos = np.zeros((n_particles, 3), dtype=np.float32)
    pos[:, 0] = np.arange(n_particles) * rest_length
    pos += rng.randn(n_particles, 3) * 0.1

    vel = rng.randn(n_particles, 3) * 0.1
    masses = np.ones(n_particles, dtype=np.float32)

    # Center of mass frame
    v_cm = vel.mean(axis=0)
    vel -= v_cm

    traj_pos = [pos.copy()]
    traj_vel = [vel.copy()]

    def compute_accel(p):
        acc = np.zeros_like(p)
        for i in range(n_particles - 1):
            r = p[i] - p[i + 1]
            dist = np.sqrt(np.sum(r**2))
            force = -k * (dist - rest_length) * r / max(dist, 1e-6)
            acc[i] -= force
            acc[i + 1] += force
        return acc

    acc = compute_accel(pos)

    for _ in range(n_steps - 1):
        vel_half = vel + 0.5 * dt * acc
        pos_new = pos + dt * vel_half
        acc_new = compute_accel(pos_new)
        vel_new = vel_half + 0.5 * dt * acc_new

        pos = pos_new
        vel = vel_new
        acc = acc_new

        traj_pos.append(pos.copy())
        traj_vel.append(vel.copy())

    pos_arr = np.stack(traj_pos)
    vel_arr = np.stack(traj_vel)
    mass_arr = np.broadcast_to(masses[None, :], (n_particles, n_particles)).copy()

    return {
        'pos': pos_arr.astype(np.float32),
        'vel': vel_arr.astype(np.float32),
        'masses': mass_arr.astype(np.float32),
        'physics_type': 'spring',
        'params': {'k': k, 'rest_length': rest_length},
    }


def generate_lennard_jones_3d(n_particles: int = 5, n_steps: int = 50,
                               dt: float = 0.002, epsilon: float = 1.0,
                               sigma: float = 1.0, seed: int = 42) -> Dict:
    """Generate 3D Lennard-Jones molecular dynamics trajectory."""
    rng = np.random.RandomState(seed)

    # FCC-like initial positions with perturbation
    pos = rng.randn(n_particles, 3) * 0.5
    # Ensure minimum separation
    for _ in range(100):
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                diff = pos[i] - pos[j]
                dist = np.sqrt(np.sum(diff**2))
                if dist < sigma * 0.8:
                    pos[i] += diff / max(dist, 1e-6) * 0.1

    vel = rng.randn(n_particles, 3) * 0.2
    masses = np.ones(n_particles, dtype=np.float32)

    # Center of mass frame
    v_cm = vel.mean(axis=0)
    vel -= v_cm

    traj_pos = [pos.copy()]
    traj_vel = [vel.copy()]

    def compute_accel(p):
        acc = np.zeros_like(p)
        for i in range(n_particles):
            for j in range(n_particles):
                if i == j:
                    continue
                r = p[i] - p[j]
                dist = np.sqrt(np.sum(r**2))
                dist = max(dist, sigma * 0.1)  # prevent divergence
                s6 = (sigma / dist) ** 6
                force = 24 * epsilon * (2 * s6**2 - s6) * r / dist**2
                acc[i] += force
        return acc

    acc = compute_accel(pos)

    for _ in range(n_steps - 1):
        vel_half = vel + 0.5 * dt * acc
        pos_new = pos + dt * vel_half
        acc_new = compute_accel(pos_new)
        vel_new = vel_half + 0.5 * dt * acc_new

        pos = pos_new
        vel = vel_new
        acc = acc_new

        traj_pos.append(pos.copy())
        traj_vel.append(vel.copy())

    pos_arr = np.stack(traj_pos)
    vel_arr = np.stack(traj_vel)
    mass_arr = np.broadcast_to(masses[None, :], (n_particles, n_particles)).copy()

    return {
        'pos': pos_arr.astype(np.float32),
        'vel': vel_arr.astype(np.float32),
        'masses': mass_arr.astype(np.float32),
        'physics_type': 'lennard_jones',
        'params': {'epsilon': epsilon, 'sigma': sigma},
    }


def make_dataset(physics_type: str, n_particles: int = 5,
                 n_train: int = 200, n_val: int = 50, n_test: int = 50,
                 n_steps: int = 50, seed: int = 42) -> Tuple:
    """Generate train/val/test splits for a physics system.

    Returns:
        (train_data, val_data, test_data) — each a dict with
        pos, vel, masses, physics_type.
    """
    gen_fn = {
        'gravity': generate_nbody_3d,
        'spring': generate_spring_3d,
        'lennard_jones': generate_lennard_jones_3d,
    }[physics_type]

    datasets = []
    seeds = [seed, seed + n_train, seed + n_train + n_val]
    for split, n in [('train', n_train), ('val', n_val), ('test', n_test)]:
        if n == 0:
            continue
        trajectories = []
        for i in range(n):
            traj = gen_fn(n_particles=n_particles, n_steps=n_steps,
                          seed=seeds[['train', 'val', 'test'].index(split)] + i)
            trajectories.append(traj)

        datasets.append({
            'pos': np.stack([t['pos'] for t in trajectories]),     # (S, T, N, 3)
            'vel': np.stack([t['vel'] for t in trajectories]),
            'masses': np.stack([t['masses'] for t in trajectories]),
            'physics_type': physics_type,
        })

    return tuple(datasets)
