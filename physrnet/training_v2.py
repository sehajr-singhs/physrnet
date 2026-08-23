"""Training loop for PSN-1 v2.

Supports:
  * Standard training with PINN + conservation losses
  * Ablation: no-PINN, no-equivariance, reasoning-only, equivariant-only
  * Rollout evaluation (multi-step prediction)
  * Equivariance verification
  * Conservation law discovery evaluation
"""

from __future__ import annotations

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _traj_to_torch(traj_dict, device='cpu'):
    """Convert trajectory dict to torch tensors."""
    pos = torch.tensor(traj_dict['pos'], dtype=torch.float32, device=device)
    vel = torch.tensor(traj_dict['vel'], dtype=torch.float32, device=device)
    masses = torch.tensor(traj_dict['masses'], dtype=torch.float32, device=device)
    return pos, vel, masses


def train_psn1v2(model, train_data, val_data, n_epochs=60, lr=1e-3,
                 w_pinn=0.5, w_conservation=0.3, batch_size=32,
                 device='cpu', out_dir='results', name='psn1v2',
                 freeze_gate=None):
    """Train PSN-1 v2 model.

    Args:
        model: PSN1v2 instance
        train_data: dict with pos (S, T, N, 3), vel, masses
        val_data: same structure
        n_epochs: training epochs
        lr: learning rate
        w_pinn: weight for PINN conservation loss
        w_conservation: weight for conservation discovery loss
        batch_size: mini-batch size
        device: torch device
        out_dir: directory for results
        name: experiment name
        freeze_gate: if set, freeze gate to this value (for ablations)

    Returns:
        results dict with metrics
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    if freeze_gate is not None:
        model.freeze_gate = freeze_gate
        model = model.to(device)
    else:
        model = model.to(device)

    # Prepare data: use timesteps 0..T-2 as input, 1..T-1 as target
    train_pos, train_vel, train_mass_raw = _traj_to_torch(train_data, device)
    val_pos, val_vel, val_mass_raw = _traj_to_torch(val_data, device)

    S_train, T_train, N, D = train_pos.shape
    S_val = val_pos.shape[0]

    # Masses: (S, N, N) -> per-particle: diagonal or mean -> (S, N)
    # Use first row of mass matrix as per-particle masses
    if train_mass_raw.dim() == 3:  # (S, N, N)
        train_mass = train_mass_raw[:, 0, :]  # (S, N)
        val_mass = val_mass_raw[:, 0, :]      # (S, N)
    else:  # already (S, N)
        train_mass = train_mass_raw
        val_mass = val_mass_raw

    # Flatten to (S*(T-1), N, 3) pairs
    n_pairs = T_train - 1
    train_in_pos = train_pos[:, :-1].reshape(-1, N, D)
    train_in_vel = train_vel[:, :-1].reshape(-1, N, D)
    train_tgt_pos = train_pos[:, 1:].reshape(-1, N, D)
    train_tgt_vel = train_vel[:, 1:].reshape(-1, N, D)
    # Masses: repeat for each timestep pair -> (S*(T-1), N)
    train_mass_flat = train_mass.unsqueeze(1).expand(-1, n_pairs, -1).reshape(-1, N)

    val_n_pairs = val_pos.shape[1] - 1
    val_in_pos = val_pos[:, :-1].reshape(-1, N, D)
    val_in_vel = val_vel[:, :-1].reshape(-1, N, D)
    val_tgt_pos = val_pos[:, 1:].reshape(-1, N, D)
    val_tgt_vel = val_vel[:, 1:].reshape(-1, N, D)
    val_mass_flat = val_mass.unsqueeze(1).expand(-1, val_n_pairs, -1).reshape(-1, N)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = {'train_loss': [], 'val_loss': [], 'val_mse': [],
               'val_energy_err': [], 'val_equiv_err': [], 'val_drift': [],
               'gate_mean': [], 'wall_time': []}

    t_start = time.time()

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Mini-batch training
        n_samples = train_in_pos.shape[0]
        perm = torch.randperm(n_samples, device=device)

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            pos_in = train_in_pos[idx]
            vel_in = train_in_vel[idx]
            pos_tgt = train_tgt_pos[idx]
            vel_tgt = train_tgt_vel[idx]
            m = train_mass_flat[idx]

            # Forward
            accel_pred, attn_w, edge_w, gate = model(pos_in, vel_in, m)

            # Predict next state (semi-implicit Euler)
            dt = 0.01
            pos_pred = pos_in + vel_in * dt + 0.5 * accel_pred * dt ** 2
            vel_pred = vel_in + accel_pred * dt

            # Data loss
            data_loss = (pos_pred - pos_tgt).pow(2).mean() + \
                       (vel_pred - vel_tgt).pow(2).mean()

            # PINN loss
            pinn = model.pinn_loss(pos_pred, vel_pred, pos_tgt, vel_tgt, m)

            loss = data_loss + w_pinn * pinn

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        history['train_loss'].append(avg_train_loss)
        history['gate_mean'].append(gate.mean().item())

        # Validation
        model.eval()
        with torch.no_grad():
            accel_pred, attn_w, edge_w, gate = model(
                val_in_pos, val_in_vel, val_mass_flat)

            dt = 0.01
            pos_pred = val_in_pos + val_in_vel * dt + 0.5 * accel_pred * dt ** 2
            vel_pred = val_in_vel + accel_pred * dt

            val_mse = (pos_pred - val_tgt_pos).pow(2).mean().item()
            pinn = model.pinn_loss(pos_pred, vel_pred, val_tgt_pos, val_tgt_vel,
                                   val_mass_flat)

            # Equivariance check
            equiv_err = _check_equivariance(model, N, D, val_mass_flat[:1], device)

            # Energy drift over rollout (use first timestep only)
            drift = _rollout_drift(model, val_pos[:2, 0], val_vel[:2, 0],
                                   val_mass[:2], dt=dt, n_steps=20, device=device)

            history['val_loss'].append((val_mse + w_pinn * pinn.item()))
            history['val_mse'].append(val_mse)
            history['val_energy_err'].append(pinn.item())
            history['val_equiv_err'].append(equiv_err)
            history['val_drift'].append(drift)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: "
                  f"train={avg_train_loss:.4f} val_mse={val_mse:.4f} "
                  f"equiv={equiv_err:.2e} drift={drift:.4f} "
                  f"gate={gate.mean().item():.3f}")

    wall_time = time.time() - t_start
    history['wall_time'].append(wall_time)

    # Final evaluation
    results = {
        'name': name,
        'n_epochs': n_epochs,
        'w_pinn': w_pinn,
        'w_conservation': w_conservation,
        'freeze_gate': freeze_gate,
        'test_mse': history['val_mse'][-1],
        'test_energy_err': history['val_energy_err'][-1],
        'test_equivariance_err': history['val_equiv_err'][-1],
        'test_drift': history['val_drift'][-1],
        'mean_gate': history['gate_mean'][-1],
        'wall_time_s': wall_time,
        'history': history,
    }

    # Save
    with open(f'{out_dir}/{name}.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _check_equivariance(model, N, D, masses, device, n_trials=5):
    """Check E(3) equivariance: rotating input should rotate output."""
    model.eval()
    total_err = 0.0

    for _ in range(n_trials):
        # Random rotation matrix (3D)
        angle = np.random.uniform(0, 2 * np.pi)
        axis = np.random.randn(3)
        axis /= np.linalg.norm(axis)

        # Rodrigues' rotation formula
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        R = torch.tensor(R, dtype=torch.float32, device=device)

        # Random positions and velocities
        pos = torch.randn(1, N, 3, device=device)
        vel = torch.randn(1, N, 3, device=device)

        # Original prediction
        accel_orig, _, _, _ = model(pos, vel, masses[:1])

        # Rotated prediction
        pos_rot = pos @ R.T
        vel_rot = vel @ R.T
        accel_rot, _, _, _ = model(pos_rot, vel_rot, masses[:1])

        # Expected: rotated output = R @ original output
        accel_expected = accel_orig @ R.T

        err = (accel_rot - accel_expected).pow(2).mean().sqrt().item()
        total_err += err

    return total_err / n_trials


def _rollout_drift(model, init_pos, init_vel, masses, dt=0.01,
                   n_steps=20, device='cpu'):
    """Compute energy drift over a multi-step rollout."""
    from .pinn import PhysicsResidual

    pos = init_pos.clone().to(device)
    vel = init_vel.clone().to(device)
    m = masses[:pos.shape[0]].to(device)

    # Determine physics type from model
    physics_type = model.pinn.physics_type
    pinn = PhysicsResidual(physics_type,
                           **({} if physics_type != 'gravity' else {'G': 1.0, 'softening': 0.5}))

    E0 = pinn.total_energy(pos, vel, m).mean().item()

    model.eval()
    with torch.no_grad():
        for _ in range(n_steps):
            accel, _, _, _ = model(pos, vel, m)
            vel = vel + accel * dt
            pos = pos + vel * dt

    E1 = pinn.total_energy(pos, vel, m).mean().item()
    return abs(E1 - E0) / max(abs(E0), 1e-8)


def run_ablation_study(train_data, val_data, n_particles=5, n_epochs=40,
                       device='cpu', out_dir='results'):
    """Run ablation study: compare full model vs. variants.

    Ablations:
      1. Full model (E3 + attention + PINN + conservation)
      2. No PINN (w_pinn=0)
      3. Equivariant-only (gate=1.0)
      4. Reasoning-only (gate=0.0)
      5. No conservation discovery
    """
    from .model_v2 import PSN1v2

    results = {}

    configs = [
        ('full', {}),
        ('no_pinn', {'w_pinn': 0.0}),
        ('equiv_only', {'freeze_gate': 1.0}),
        ('reason_only', {'freeze_gate': 0.0}),
        ('no_conservation', {}),
    ]

    for name, kwargs in configs:
        print(f"\n=== Ablation: {name} ===")
        use_conservation = name != 'no_conservation'
        model = PSN1v2(n_particles=n_particles, physics_type='gravity',
                       use_conservation=use_conservation)

        freeze = kwargs.get('freeze_gate', None)
        w_pinn = kwargs.get('w_pinn', 0.5)

        r = train_psn1v2(model, train_data, val_data,
                         n_epochs=n_epochs, w_pinn=w_pinn,
                         device=device, out_dir=out_dir,
                         name=f'ablation_{name}',
                         freeze_gate=freeze)
        results[name] = r

    return results
