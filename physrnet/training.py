"""Training and evaluation loops for PhysRNet.

The objective is per-particle acceleration MSE plus physics-informed
conservation residuals (energy, linear and angular momentum). Evaluation
reports five quantities beyond raw MSE:

  * rotation-equivariance error   — the model must be exactly E(2)-equivariant
  * rollout energy/momentum drift — long-horizon stability of the learnt model
  * interaction-graph recovery    — do the learnt edge weights match the true
                                    interaction topology?
  * generalization (unseen masses/initial conditions)
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr, mannwhitneyu

from .pinn import PhysicsResidual


def _flatten(data: dict):
    """Flatten (n_traj, n_steps, N, D) -> (n_traj * n_steps, N, D)."""
    return {
        "pos": torch.tensor(data["positions"], dtype=torch.float32).reshape(-1, *data["positions"].shape[2:]),
        "vel": torch.tensor(data["velocities"], dtype=torch.float32).reshape(-1, *data["velocities"].shape[2:]),
        "acc": torch.tensor(data["accelerations"], dtype=torch.float32).reshape(-1, *data["accelerations"].shape[2:]),
        "masses": torch.tensor(data["masses"], dtype=torch.float32).repeat_interleave(data["positions"].shape[1], dim=0),
    }


def _batches(data: dict, batch_size: int, rng: np.random.Generator):
    """Yield shuffled mini-batches of (state, target) pairs."""
    flat = _flatten(data)
    n = flat["pos"].shape[0]
    perm = rng.permutation(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield {k: v[idx] for k, v in flat.items()}


def train_physrnet(model, train_data: dict, val_data: dict,
                   n_epochs: int = 60, lr: float = 2e-3,
                   batch_size: int = 256, w_physics: float = 0.5,
                   device: str = "cpu", out_dir: str = "results",
                   dt: float = 0.01) -> dict:
    """Train the model and return the training history."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    rng = np.random.default_rng(0)

    history = {"train_loss": [], "train_mse": [], "train_phys": [],
               "val_mse": [], "val_energy_err": []}
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train()
        tot_loss = tot_mse = tot_phys = 0.0
        n_batch = 0
        for batch in _batches(train_data, batch_size, rng):
            pos = batch["pos"].to(device)
            vel = batch["vel"].to(device)
            masses = batch["masses"].to(device)
            acc_true = batch["acc"].to(device)

            accel_pred, edge_weights, gate = model(pos, vel, masses)

            mse = nn.functional.mse_loss(accel_pred, acc_true)

            # semi-implicit Euler step to the next state for the physics residual
            vel_next = vel + accel_pred * dt
            pos_next = pos + vel_next * dt
            vel_true_next = vel + acc_true * dt
            pos_true_next = pos + vel_true_next * dt
            phys = model.physics_loss(pos_next, vel_next,
                                      pos_true_next, vel_true_next, masses)

            loss = mse + w_physics * phys
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tot_loss += loss.item()
            tot_mse += mse.item()
            tot_phys += phys.item()
            n_batch += 1

        scheduler.step()
        history["train_loss"].append(tot_loss / max(1, n_batch))
        history["train_mse"].append(tot_mse / max(1, n_batch))
        history["train_phys"].append(tot_phys / max(1, n_batch))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            m = evaluate_physrnet(model, val_data, device, dt=dt)
            history["val_mse"].append(m["mse"])
            history["val_energy_err"].append(m["energy_err"])
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1:3d}: loss={history['train_loss'][-1]:.5f} "
                      f"mse={history['train_mse'][-1]:.5f} "
                      f"val_mse={m['mse']:.6f} energy_err={m['energy_err']:.6f}")

    history["wall_time_s"] = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history


@torch.no_grad()
def evaluate_physrnet(model, data: dict, device: str = "cpu", dt: float = 0.01,
                      n_eval: int = 400) -> dict:
    """Evaluate MSE, energy error, equivariance, drift, and graph recovery."""
    model.eval()
    flat = _flatten(data)
    n = min(n_eval, flat["pos"].shape[0])
    pos = flat["pos"][:n].to(device)
    vel = flat["vel"][:n].to(device)
    acc = flat["acc"][:n].to(device)
    masses = flat["masses"][:n].to(device)

    accel_pred, edge_weights, gate = model(pos, vel, masses)

    out = {"mse": nn.functional.mse_loss(accel_pred, acc).item()}

    # energy error at the next step (semi-implicit Euler)
    physics = model.physics
    vel_next = vel + accel_pred * dt
    pos_next = pos + vel_next * dt
    vel_true_next = vel + acc * dt
    pos_true_next = pos + vel_true_next * dt
    E_pred = physics.total_energy(pos_next, vel_next, masses)
    E_true = physics.total_energy(pos_true_next, vel_true_next, masses)
    out["energy_err"] = (E_pred - E_true).pow(2).mean().item()

    # momentum error
    p_pred = physics.momentum(vel_next, masses)
    p_true = physics.momentum(vel_true_next, masses)
    out["momentum_err"] = (p_pred - p_true).pow(2).mean().item()

    # rotation equivariance: R(a(x)) vs a(R(x)) for a random rotation
    out["equivariance_err"] = _rotation_equivariance_error(model, pos, vel, masses)

    # long-horizon rollout drift
    out.update(_rollout_drift(model, pos, vel, masses, physics, dt))

    # interaction-graph recovery (compare learned edge weights to true forces)
    out["graph_recovery"] = _graph_recovery(edge_weights, pos, masses, model.physics)

    out["mean_gate"] = gate.mean().item()
    return out


@torch.no_grad()
def _rotation_equivariance_error(model, pos, vel, masses, n_rots: int = 8) -> float:
    """||R·a(x) - a(R·x)||² averaged over random rotations. 0 => exactly equivariant."""
    B, N, D = pos.shape
    total = 0.0
    for _ in range(n_rots):
        theta = torch.rand(1, device=pos.device) * 2 * math.pi
        R = torch.tensor([[theta.cos(), -theta.sin()],
                          [theta.sin(), theta.cos()]], device=pos.device)
        pos_r = pos @ R
        vel_r = vel @ R
        a0, _, _ = model(pos, vel, masses)
        a1, _, _ = model(pos_r, vel_r, masses)
        total += (a0 @ R - a1).pow(2).mean().item()
    return total / n_rots


@torch.no_grad()
def _rollout_drift(model, pos, vel, masses, physics, dt, n_steps: int = 50) -> dict:
    """Energy/momentum drift when the learnt model is rolled out as a simulator."""
    B, N, D = pos.shape
    pos_r, vel_r = pos.clone(), vel.clone()
    E0 = physics.total_energy(pos_r, vel_r, masses).mean().item()
    P0 = physics.momentum(vel_r, masses).norm(dim=-1).mean().item()
    for _ in range(n_steps):
        a, _, _ = model(pos_r, vel_r, masses)
        vel_r = vel_r + a * dt
        pos_r = pos_r + vel_r * dt
    E1 = physics.total_energy(pos_r, vel_r, masses).mean().item()
    P1 = physics.momentum(vel_r, masses).norm(dim=-1).mean().item()
    return {
        "rollout_steps": n_steps,
        "energy_drift": abs(E1 - E0),
        "momentum_drift": abs(P1 - P0),
    }


@torch.no_grad()
def _graph_recovery(edge_weights: torch.Tensor, pos, masses, physics) -> dict:
    """Interactability of the recovered interaction graph (vectorized).

    * topology_recall (spring): do the top-|E_true| learned edge weights match
      the true chain adjacency — i.e. does the model learn *which* pairs couple?
    * strength_corr (all systems): Spearman rank correlation between the learned
      signed edge weights w_ij and the *true* pairwise force w_ij — i.e. does the
      model recover the actual interaction strengths, not just the topology?
    """
    B, N, D = pos.shape
    ptype = physics.physics_type
    pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
    E = len(pairs)

    w_all = edge_weights.detach().reshape(B, E).cpu().numpy()   # (B, E) signed
    pb = pos.cpu().numpy()                                      # (B, N, D)
    mb = masses.cpu().numpy()                                   # (B, N)

    # true pairwise force magnitudes (vectorized over pairs and batch)
    rij = pb[:, np.array([p[0] for p in pairs]), :] - pb[:, np.array([p[1] for p in pairs]), :]
    r = np.linalg.norm(rij, axis=-1)                            # (B, E)
    mj = mb[:, np.array([p[1] for p in pairs])]                 # (B, E) source mass

    if ptype == "gravity":
        w_true = -physics.G * mj * r / (r ** 2 + physics.softening ** 2) ** 1.5
    elif ptype == "spring":
        adj = np.array([1.0 if abs(i - j) == 1 else 0.0 for (i, j) in pairs])
        w_true = -physics.k * (r - physics.rest_length) * adj[None, :]
    elif ptype == "lennard_jones":
        r = np.clip(r, 0.3, None)
        s6 = (physics.sigma / r) ** 6
        w_true = 24.0 * physics.epsilon * (2.0 * s6 ** 2 - s6) / r
    else:
        w_true = np.zeros_like(r)

    # per-sample Spearman correlation between learned and true weights
    corr_scores = []
    for b in range(B):
        if np.all(w_true[b] == 0.0) or np.all(w_all[b] == w_all[b][0]):
            continue
        c = spearmanr(w_all[b], w_true[b]).correlation
        if np.isfinite(c):
            corr_scores.append(c)

    # spring topology recovery: (a) top-k recall, (b) adjacency ROC-AUC of |w|
    topo_scores, auc_scores = [], []
    if ptype == "spring":
        adj_mask = np.array([1.0 if abs(i - j) == 1 else 0.0 for (i, j) in pairs])
        true_pairs = {(i, i + 1) for i in range(N - 1)} | {(i + 1, i) for i in range(N - 1)}
        k = len(true_pairs)
        for b in range(B):
            abs_w = np.abs(w_all[b])
            top = np.argsort(-abs_w)[:k]
            top_pairs = {pairs[int(t)] for t in top}
            topo_scores.append(len(top_pairs & true_pairs) / k)

            pos_w = abs_w[adj_mask == 1.0]
            neg_w = abs_w[adj_mask == 0.0]
            if pos_w.size and neg_w.size:
                u, _ = mannwhitneyu(pos_w, neg_w, alternative="two-sided")
                auc_scores.append(u / (pos_w.size * neg_w.size))

    return {
        "topology_recall": float(np.mean(topo_scores)) if topo_scores else None,
        "adjacency_auc": float(np.mean(auc_scores)) if auc_scores else None,
        "strength_corr": float(np.mean(corr_scores)) if corr_scores else float("nan"),
    }
