#!/usr/bin/env python3
"""PSN-1 Universal training on Modal GPU.

Training PSNUniversal across 9 physics domains on a T4 GPU.
The training runs detached; check status at https://modal.com/apps/sehajr-singhs/psn1-universal

Usage:
    modal run physrnet/modal_train_universal.py --n-epochs 20
    modal run --detach physrnet/modal_train_universal.py --n-epochs 20
"""

import json
import os
import time

import modal

PHYSRNET_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("psn1-universal")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "numpy")
    .add_local_dir(PHYSRNET_DIR, remote_path="/root/physrnet",
                   ignore=lambda p: "__pycache__" in str(p) or str(p).endswith(".pyc"))
)


@app.function(gpu="T4", timeout=1800, image=image)
def train(n_epochs: int = 20):
    import sys

    sys.path.insert(0, "/root/physrnet")
    import numpy as np
    import torch
    import torch.nn as nn

    from physrnet.model_universal import PSNUniversal
    from physrnet.conservation import PhysicsDiscovery
    from physrnet.pinn import PhysicsResidual

    # ── config ──
    DOMAIN_CONFIGS = {
        "gravity": {"n_particles": 4, "n_steps": 50, "dt": 0.01},
        "spring": {"n_particles": 4, "n_steps": 50, "dt": 0.005},
        "lennard_jones": {"n_particles": 4, "n_steps": 50, "dt": 0.002},
        "fluid": {"n_particles": 64, "n_steps": 30, "dt": 0.01},
        "electromagnetism": {"n_particles": 8, "n_steps": 50, "dt": 0.005},
        "quantum": {"n_particles": 64, "n_steps": 40, "dt": 0.005},
        "heat": {"n_particles": 32, "n_steps": 30, "dt": 0.01},
        "relativistic": {"n_particles": 4, "n_steps": 50, "dt": 0.01},
        "thermo_ideal": {"n_particles": 1, "n_steps": 50, "dt": 0.01},
    }
    domains = list(DOMAIN_CONFIGS.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LR = 1e-3
    W_PINN = 0.3
    HIDDEN = 64
    N_SCALAR = 8
    N_HEADS = 4
    DOM_EMB = 16
    N_TRAIN = 40
    N_VAL = 5

    # ── synthetic data ──
    def gen(domain, n_samples, seed):
        cfg = DOMAIN_CONFIGS[domain]
        rng = np.random.RandomState(seed)
        omega_map = {
            "gravity": 0.5, "spring": 1.5, "lennard_jones": 2.0,
            "fluid": 0.3, "electromagnetism": 0.8, "quantum": 1.2,
            "heat": 0.1, "relativistic": 0.6, "thermo_ideal": 0.2,
        }
        omega = omega_map.get(domain, 1.0)
        damping = 0.05; dt = 0.05
        N = cfg["n_particles"]; T = cfg["n_steps"]; D = 3
        pos = np.zeros((n_samples, T, N, D), dtype=np.float32)
        vel = np.zeros((n_samples, T, N, D), dtype=np.float32)
        for i in range(n_samples):
            amp = rng.uniform(0.5, 2.0, (N, D))
            phase = rng.uniform(0, 2 * np.pi, (N, D))
            for t in range(T):
                tv = t * dt; env = np.exp(-damping * omega * tv)
                pos[i, t] = amp * env * np.cos(omega * tv + phase)
                vel[i, t] = -amp * env * omega * np.sin(omega * tv + phase)
        masses = np.ones((n_samples, T, N), dtype=np.float32)
        return {"pos": pos, "vel": vel, "masses": masses}

    print("=" * 60)
    print(f"  PSN-1 Modal GPU: {len(domains)} domains, {n_epochs} epochs")
    print(f"  Device: {device}")
    print("=" * 60)

    t0 = time.time()
    print("\nGenerating data...")
    train_data, val_data = {}, {}
    for d in domains:
        train_data[d] = (gen(d, N_TRAIN, 42), DOMAIN_CONFIGS[d])
        val_data[d] = (gen(d, N_VAL, 999), DOMAIN_CONFIGS[d])
        print(f"  {d:20s} train={train_data[d][0]['pos'].shape}")

    # ── prepare batches ──
    all_batches = []
    for d in domains:
        data, cfg = train_data[d]
        pos = torch.tensor(data["pos"], dtype=torch.float32, device=device)
        vel = torch.tensor(data["vel"], dtype=torch.float32, device=device)
        mass = torch.tensor(data["masses"], dtype=torch.float32, device=device)
        S_, T_, N_, D_ = pos.shape
        in_pos = pos[:, :-1].reshape(-1, N_, D_)
        in_vel = vel[:, :-1].reshape(-1, N_, D_)
        m = mass[:, : T_ - 1].reshape(-1, N_)
        tgt_pos = pos[:, 1:].reshape(-1, N_, D_)
        tgt_vel = vel[:, 1:].reshape(-1, N_, D_)
        n_pairs = in_pos.shape[0]
        for start in range(0, n_pairs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_pairs)
            all_batches.append({
                "pos": in_pos[start:end], "vel": in_vel[start:end],
                "masses": m[start:end],
                "pos_tgt": tgt_pos[start:end], "vel_tgt": tgt_vel[start:end],
                "domain": d, "dt": cfg["dt"],
            })
    print(f"  {len(all_batches)} batches (bs={BATCH_SIZE})")

    # ── model ──
    model = PSNUniversal(
        n_particles=max(c["n_particles"] for c in DOMAIN_CONFIGS.values()),
        hidden=HIDDEN, n_scalar=N_SCALAR, n_heads=N_HEADS,
        n_domains=len(domains), domain_emb_dim=DOM_EMB,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    history = {"train_loss": [], "per_domain_val_mse": {}}

    print(f"\nTraining {n_epochs} epochs...")
    for epoch in range(n_epochs):
        te = time.time()
        np.random.shuffle(all_batches)
        model.train()
        total_loss = 0.0
        for batch in all_batches:
            pos, vel, masses = batch["pos"], batch["vel"], batch["masses"]
            domain, dt = batch["domain"], batch["dt"]
            pred, aux = model(pos, vel, masses, domain)
            accel = pred["acceleration"]
            pos_p = pos + vel * dt + 0.5 * accel * dt**2
            vel_p = vel + accel * dt
            data_loss = (pos_p - batch["pos_tgt"]).pow(2).mean() + \
                        (vel_p - batch["vel_tgt"]).pow(2).mean()
            try:
                pinn_loss = model.physics_loss(
                    {"acceleration": accel, "pos_pred": pos_p, "vel_pred": vel_p},
                    {"pos": pos, "vel": vel, "masses": masses, "dt": dt}, domain)
                if not isinstance(pinn_loss, torch.Tensor):
                    pinn_loss = torch.tensor(0.0, device=device)
                if pinn_loss.dim() > 0:
                    pinn_loss = pinn_loss.mean()
            except Exception:
                pinn_loss = torch.tensor(0.0, device=device)
            loss = data_loss + W_PINN * pinn_loss
            gate = aux.get("gate", torch.ones(1, device=device))
            loss = loss + 0.01 * (gate * (1 - gate)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()
        avg_loss = total_loss / max(len(all_batches), 1)
        history["train_loss"].append(avg_loss)
        if (epoch + 1) % 3 == 0 or epoch == 0:
            dt_epoch = time.time() - te
            print(f"  Epoch {epoch+1:3d}: loss={avg_loss:.4f}  [{dt_epoch:.0f}s]")

    # ── final eval ──
    print("\nFinal evaluation:")
    final = {}
    with torch.no_grad():
        model.eval()
        for d in domains:
            data, cfg = val_data[d]
            pos = torch.tensor(data["pos"], dtype=torch.float32, device=device)
            vel = torch.tensor(data["vel"], dtype=torch.float32, device=device)
            mass = torch.tensor(data["masses"], dtype=torch.float32, device=device)
            dt = cfg["dt"]
            pred, aux = model(pos[:, 0], vel[:, 0], mass[:, 0, :], d)
            accel = pred["acceleration"]
            pos_p = pos[:, 0] + vel[:, 0] * dt + 0.5 * accel * dt**2
            mse = (pos_p - pos[:, 1]).pow(2).mean().item()
            gate = aux.get("gate", torch.tensor(0.5)).mean().item()
            final[d] = {"mse": mse, "gate": gate}
            print(f"  {d:22s} mse={mse:.6e} gate={gate:.3f}")

    total_time = time.time() - t0
    mean_mse = float(np.mean([v["mse"] for v in final.values()]))
    result = {
        "n_domains": len(domains), "n_epochs": n_epochs,
        "n_params": n_params, "device": device, "batch_size": BATCH_SIZE,
        "final_mean_mse": mean_mse, "per_domain": final,
        "train_loss_final": float(history["train_loss"][-1]),
        "total_time_s": float(total_time),
    }
    print(f"\n  Mean MSE: {mean_mse:.6e}")
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print("=" * 60)
    return result


@app.local_entrypoint()
def main(n_epochs: int = 20):
    """Run PSN-1 universal training with --n-epochs (default 20).

    Use --detach to run in the background on Modal's GPU:
        modal run --detach physrnet/modal_train_universal.py --n-epochs 15
    """
    result = train.remote(n_epochs)
    print("\nResult:", json.dumps(result, indent=2))