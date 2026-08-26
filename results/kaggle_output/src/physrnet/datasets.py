"""Synthetic physics datasets for training and evaluation.

Trajectories are generated on-the-fly with vectorized RK4 integration — all
trajectories are advanced simultaneously, so generation is fast even for
large datasets. Each trajectory is a sequence of (position, velocity)
snapshots together with the true accelerations (targets) and masses (inputs).

Datasets (all internally-forced, so energy/momentum are conserved):
  * NBodyGravity : inverse-square gravitational attraction, F = -G m_i m_j / r^2
  * SpringChain  : nearest-neighbour Hooke springs,  F = -k (r - r_0)
  * LennardJones : 6-12 potential particles,           F = -grad V_LJ(r)
"""

from __future__ import annotations

import numpy as np


def _rk4_step(pos, vel, masses, acc_fn, dt):
    """One vectorized RK4 step. pos, vel: (T, N, D). acc_fn: (pos, masses) -> (T, N, D)."""
    k1v = acc_fn(pos, masses)
    k1x = vel
    k2v = acc_fn(pos + 0.5 * dt * k1x, masses)
    k2x = vel + 0.5 * dt * k1v
    k3v = acc_fn(pos + 0.5 * dt * k2x, masses)
    k3x = vel + 0.5 * dt * k2v
    k4v = acc_fn(pos + dt * k3x, masses)
    k4x = vel + dt * k3v
    new_pos = pos + (dt / 6) * (k1x + 2 * k2x + 2 * k3x + k4x)
    new_vel = vel + (dt / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return new_pos, new_vel


def _rollout(pos0, vel0, masses, acc_fn, dt, n_steps):
    """Integrate a whole batch of trajectories, collecting states and accelerations."""
    T, N, D = pos0.shape
    traj_pos = np.empty((T, n_steps, N, D), dtype=np.float32)
    traj_vel = np.empty((T, n_steps, N, D), dtype=np.float32)
    traj_acc = np.empty((T, n_steps, N, D), dtype=np.float32)
    pos, vel = pos0, vel0
    for t in range(n_steps):
        acc = acc_fn(pos, masses)
        traj_pos[:, t] = pos
        traj_vel[:, t] = vel
        traj_acc[:, t] = acc
        pos, vel = _rk4_step(pos, vel, masses, acc_fn, dt)
    return traj_pos, traj_vel, traj_acc


def _pack(masses, pos_list, vel_list, acc_list):
    return {
        "positions": pos_list,        # (n_traj, n_steps, N, D)
        "velocities": vel_list,
        "accelerations": acc_list,
        "masses": masses,             # (n_traj, N)
    }


class NBodyGravity:
    """N particles under inverse-square gravity (2D).

    a_i = -G * sum_{j != i} m_j (r_i - r_j) / |r_i - r_j|^3
    """

    def __init__(self, n_particles: int = 3, G: float = 1.0, dt: float = 0.01,
                 n_steps: int = 100, softening: float = 0.5):
        self.n = n_particles
        self.G = G
        self.dt = dt
        self.n_steps = n_steps
        self.softening = softening

    def accelerations(self, pos, masses):
        """Plummer-softened gravity: a_i = -G sum_j m_j (r_i - r_j) / (r^2 + eps^2)^(3/2).

        pos: (T, N, 2), masses: (T, N) -> (T, N, 2).
        """
        diff = pos[:, :, None, :] - pos[:, None, :, :]            # (T, N, N, 2)
        r2 = (diff ** 2).sum(-1, keepdims=True) + self.softening ** 2
        acc = -self.G * (masses[:, None, :, None] * diff / r2 ** 1.5).sum(axis=2)
        return acc

    def generate(self, n_traj: int = 500, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        masses = (0.5 + rng.random((n_traj, self.n)) * 1.5).astype(np.float32)
        pos = ((rng.random((n_traj, self.n, 2)) - 0.5) * 4).astype(np.float32)
        vel = ((rng.random((n_traj, self.n, 2)) - 0.5) * 2).astype(np.float32)
        v_cm = (masses[:, :, None] * vel).sum(1) / masses.sum(1, keepdims=True)
        vel -= v_cm[:, None, :]
        p, v, a = _rollout(pos, vel, masses, self.accelerations, self.dt, self.n_steps)
        return _pack(masses, p, v, a)


class SpringChain:
    """Particles connected in a linear chain by Hooke springs (2D).

    a_i = -(k / m_i) * sum_{j neighbour} (|r_i - r_j| - L0) (r_i - r_j) / |r_i - r_j|
    """

    def __init__(self, n_particles: int = 4, k: float = 10.0, L0: float = 1.0,
                 dt: float = 0.005, n_steps: int = 100):
        self.n = n_particles
        self.k = k
        self.L0 = L0
        self.dt = dt
        self.n_steps = n_steps
        adj = np.zeros((n_particles, n_particles))
        for i in range(n_particles - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        self.adj = adj

    def accelerations(self, pos, masses):
        diff = pos[:, :, None, :] - pos[:, None, :, :]            # (T, N, N, 2)
        dist = np.linalg.norm(diff, axis=-1, keepdims=True) + 1e-6
        force = -self.k * (dist - self.L0) * diff / dist          # (T, N, N, 2)
        force *= self.adj[None, :, :, None]
        return (force / masses[:, None, :, None]).sum(axis=2)

    def generate(self, n_traj: int = 500, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        masses = np.ones((n_traj, self.n), dtype=np.float32)
        pos = np.zeros((n_traj, self.n, 2), dtype=np.float32)
        for i in range(self.n):
            pos[:, i, 0] = i * self.L0 + rng.random(n_traj) * 0.3 - 0.15
            pos[:, i, 1] = rng.random(n_traj) * 0.4 - 0.2
        vel = ((rng.random((n_traj, self.n, 2)) - 0.5) * 0.5).astype(np.float32)
        p, v, a = _rollout(pos, vel, masses, self.accelerations, self.dt, self.n_steps)
        return _pack(masses, p, v, a)


class LennardJones:
    """Particles under a 6-12 Lennard-Jones potential (2D).

    a_i = -(1/m_i) * grad_i V,  V = 4 eps sum_{i<j} [ (sigma/r)^12 - (sigma/r)^6 ]
    """

    def __init__(self, n_particles: int = 4, epsilon: float = 1.0,
                 sigma: float = 1.0, dt: float = 0.002, n_steps: int = 100,
                 r_min: float = 0.3):
        self.n = n_particles
        self.eps = epsilon
        self.sigma = sigma
        self.dt = dt
        self.n_steps = n_steps
        self.r_min = r_min

    def accelerations(self, pos, masses):
        diff = pos[:, :, None, :] - pos[:, None, :, :]            # (T, N, N, 2)
        r = np.linalg.norm(diff, axis=-1, keepdims=True)
        r = np.clip(r, self.r_min, None)                          # soften short-range blow-up
        s6 = (self.sigma / r) ** 6
        fmag = 24.0 * self.eps * (2.0 * s6 ** 2 - s6) / r         # (T, N, N, 1)
        force = fmag * diff / r                                   # (T, N, N, 2)
        eye = 1.0 - np.eye(self.n, dtype=np.float32)[None, :, :, None]
        force *= eye
        return (force / masses[:, None, :, None]).sum(axis=2)

    def generate(self, n_traj: int = 500, seed: int = 42) -> dict:
        rng = np.random.default_rng(seed)
        masses = np.ones((n_traj, self.n), dtype=np.float32)
        # particles on a ring (guaranteed separation) with jitter
        angles = 2 * np.pi * np.arange(self.n) / self.n + rng.random((n_traj, self.n)) * 0.3
        radius = 1.0 + (rng.random((n_traj, self.n)) - 0.5) * 0.2
        pos = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1)
        pos = pos.astype(np.float32)
        vel = ((rng.random((n_traj, self.n, 2)) - 0.5) * 0.5).astype(np.float32)
        p, v, a = _rollout(pos, vel, masses, self.accelerations, self.dt, self.n_steps)
        return _pack(masses, p, v, a)
