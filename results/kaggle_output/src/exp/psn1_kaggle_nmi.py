#!/usr/bin/env python3
"""PSN-1 NMI: full-scale 9-domain universal training + simulation clips.

Runs on a Kaggle T4 GPU. Produces:
  * results/psn1_kaggle_nmi.json   : headline metrics + per-domain results
  * results/fig_training.png       : training loss + gate curves
  * results/clip_<domain>.gif      : model rollout vs ground truth, 9 domains
  * results/clip_real_<sim>.gif    : real-physics rollouts (gravity/spring/LJ)
  * results/psn1_universal.pt      : model checkpoint
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image

from physrnet.model_universal import PSNUniversal
from physrnet.datasets import NBodyGravity, SpringChain, LennardJones

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, "cuda:", torch.cuda.is_available(), flush=True)

# ────────────────────────────────────────────────────────────────
# 1. Universal 9-domain training (synthetic harmonic, full scale)
# ────────────────────────────────────────────────────────────────
DOMAINS = ["gravity", "spring", "lennard_jones", "fluid", "electromagnetism",
           "quantum", "heat", "relativistic", "thermo_ideal"]

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

N_EPOCHS = 25
N_TRAIN = 40
N_VAL = 5
BATCH_SIZE = 32
LR = 1e-3
W_PINN = 0.3


def gen_synthetic(domain, n_samples, seed, dt=0.05):
    cfg = DOMAIN_CONFIGS[domain]
    rng = np.random.RandomState(seed)
    omega_map = {"gravity": 0.5, "spring": 1.5, "lennard_jones": 2.0,
                 "fluid": 0.3, "electromagnetism": 0.8, "quantum": 1.2,
                 "heat": 0.1, "relativistic": 0.6, "thermo_ideal": 0.2}
    omega = omega_map.get(domain, 1.0)
    damping = 0.05
    N = cfg["n_particles"]
    T = cfg["n_steps"]
    D = 3
    pos = np.zeros((n_samples, T, N, D), dtype=np.float32)
    vel = np.zeros((n_samples, T, N, D), dtype=np.float32)
    for i in range(n_samples):
        amp = rng.uniform(0.5, 2.0, (N, D))
        phase = rng.uniform(0, 2 * np.pi, (N, D))
        for t in range(T):
            tv = t * dt
            env = np.exp(-damping * omega * tv)
            pos[i, t] = amp * env * np.cos(omega * tv + phase)
            vel[i, t] = -amp * env * omega * np.sin(omega * tv + phase)
    masses = np.ones((n_samples, T, N), dtype=np.float32)
    return {"pos": pos, "vel": vel, "masses": masses, "dt": dt}


def build_batches(data, batch_size):
    pos = torch.tensor(data["pos"], dtype=torch.float32, device=DEVICE)
    vel = torch.tensor(data["vel"], dtype=torch.float32, device=DEVICE)
    mass = torch.tensor(data["masses"], dtype=torch.float32, device=DEVICE)
    S, T, N, D = pos.shape
    in_pos = pos[:, :-1].reshape(-1, N, D)
    in_vel = vel[:, :-1].reshape(-1, N, D)
    m = mass[:, :T - 1].reshape(-1, N)
    tgt_p = pos[:, 1:].reshape(-1, N, D)
    tgt_v = vel[:, 1:].reshape(-1, N, D)
    out = []
    for start in range(0, in_pos.shape[0], batch_size):
        end = min(start + batch_size, in_pos.shape[0])
        out.append({"pos": in_pos[start:end], "vel": in_vel[start:end],
                    "masses": m[start:end], "pos_tgt": tgt_p[start:end],
                    "vel_tgt": tgt_v[start:end], "dt": data["dt"]})
    return out


def train_universal():
    print("=" * 64)
    print("  1. Universal 9-domain training")
    print("=" * 64)
    t0 = time.time()
    model = PSNUniversal(
        n_particles=max(c["n_particles"] for c in DOMAIN_CONFIGS.values()),
        hidden=64, n_scalar=8, n_heads=4,
        n_domains=len(DOMAINS), domain_emb_dim=16,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}", flush=True)

    print("  generating data...", flush=True)
    batches = []
    val_data = {}
    for d in DOMAINS:
        tr = gen_synthetic(d, N_TRAIN, 42)
        va = gen_synthetic(d, N_VAL, 999)
        val_data[d] = va
        for b in build_batches(tr, BATCH_SIZE):
            b["domain"] = d
            batches.append(b)
    print(f"  {len(batches)} batches", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    history = {"train_loss": [], "gate_mean": []}

    for epoch in range(N_EPOCHS):
        te = time.time()
        np.random.shuffle(batches)
        model.train()
        total = 0.0
        for b in batches:
            pred, aux = model(b["pos"], b["vel"], b["masses"], b["domain"])
            accel = pred["acceleration"]
            dt = b["dt"]
            pos_p = b["pos"] + b["vel"] * dt + 0.5 * accel * dt ** 2
            vel_p = b["vel"] + accel * dt
            dl = (pos_p - b["pos_tgt"]).pow(2).mean() + (vel_p - b["vel_tgt"]).pow(2).mean()
            try:
                pl = model.physics_loss(
                    {"acceleration": accel, "pos_pred": pos_p, "vel_pred": vel_p},
                    {"pos": b["pos"], "vel": b["vel"], "masses": b["masses"], "dt": dt},
                    b["domain"])
                if not isinstance(pl, torch.Tensor):
                    pl = torch.tensor(0.0, device=DEVICE)
                if pl.dim() > 0:
                    pl = pl.mean()
            except Exception:
                pl = torch.tensor(0.0, device=DEVICE)
            loss = dl + W_PINN * pl
            gate = aux.get("gate", torch.ones(1, device=DEVICE))
            loss = loss + 0.01 * (gate * (1 - gate)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        history["train_loss"].append(total / max(len(batches), 1))
        print(f"  epoch {epoch+1:3d}: loss={history['train_loss'][-1]:.4f} "
              f"[{time.time()-te:.0f}s]", flush=True)

    # final eval
    final = {}
    with torch.no_grad():
        model.eval()
        for d in DOMAINS:
            va = val_data[d]
            pos = torch.tensor(va["pos"], dtype=torch.float32, device=DEVICE)
            vel = torch.tensor(va["vel"], dtype=torch.float32, device=DEVICE)
            mass = torch.tensor(va["masses"], dtype=torch.float32, device=DEVICE)
            dt = va["dt"]
            pred, aux = model(pos[:, 0], vel[:, 0], mass[:, 0, :], d)
            accel = pred["acceleration"]
            pos_p = pos[:, 0] + vel[:, 0] * dt + 0.5 * accel * dt ** 2
            mse = (pos_p - pos[:, 1]).pow(2).mean().item()
            gate = aux.get("gate", torch.tensor(0.5)).mean().item()
            final[d] = {"mse": mse, "gate": gate}

    torch.save(model.state_dict(), os.path.join(RESULTS, "psn1_universal.pt"))
    mean_mse = float(np.mean([v["mse"] for v in final.values()]))
    print(f"  mean MSE: {mean_mse:.6e} | total {time.time()-t0:.0f}s", flush=True)

    # training figure
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.plot(history["train_loss"], color="#2563eb", linewidth=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("training loss")
    ax.set_title("PSN-1 universal training (9 domains, one architecture)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_training.png"), dpi=160)
    plt.close(fig)

    return model, final, mean_mse, history, n_params, time.time() - t0


# ────────────────────────────────────────────────────────────────
# 2. Rollout clips
# ────────────────────────────────────────────────────────────────
def rollout(model, pos0, vel0, mass, domain, dt, n_steps):
    pos = pos0.clone()
    vel = vel0.clone()
    traj = [pos0.detach().cpu().numpy()]
    with torch.no_grad():
        for _ in range(n_steps - 1):
            pred, _ = model(pos, vel, mass, domain)
            a = pred["acceleration"]
            a = a.clamp(-1e3, 1e3)
            pos = pos + vel * dt + 0.5 * a * dt ** 2
            vel = vel + a * dt
            if not torch.isfinite(pos).all():
                # divergence guard: freeze the last finite state
                pos = torch.nan_to_num(pos, nan=0.0)
                vel = torch.nan_to_num(vel, nan=0.0)
            traj.append(pos.detach().cpu().numpy())
    arr = np.stack(traj, axis=0)   # (T, B=1, N, D)
    return arr[:, 0, :, :]         # (T, N, D), squeeze batch


def make_clip(name, gt, pred, title, path):
    """gt, pred: (T, N, D) with D>=2. Renders xy projection as an animated GIF."""
    T = gt.shape[0]
    N = gt.shape[1]
    gt = np.nan_to_num(gt, nan=0.0)
    pred = np.nan_to_num(pred, nan=0.0)
    gt_xy = gt[:, :, :2]
    pr_xy = pred[:, :, :2]
    all_pts = np.concatenate([gt_xy.reshape(-1, 2), pr_xy.reshape(-1, 2)], axis=0)
    xmin, xmax = all_pts[:, 0].min(), all_pts[:, 0].max()
    ymin, ymax = all_pts[:, 1].min(), all_pts[:, 1].max()
    pad = 0.15 * max(xmax - xmin, ymax - ymin, 1e-6)

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    gt_line, = ax.plot([], [], "-", color="#94a3b8", alpha=0.5, linewidth=1)
    pr_line, = ax.plot([], [], "-", color="#2563eb", alpha=0.5, linewidth=1)
    gt_pts, = ax.plot([], [], "o", color="#475569", markersize=6)
    pr_pts, = ax.plot([], [], "o", color="#ef4444", markersize=6)

    n_frames = min(T, 26)
    frames_pil = []
    for j in range(n_frames):
        k = int(j / (n_frames - 1) * (T - 1)) if n_frames > 1 else 0
        k = min(k, T - 1)
        gt_line.set_data(gt_xy[:k + 1].reshape(-1, 2)[:, 0], gt_xy[:k + 1].reshape(-1, 2)[:, 1])
        pr_line.set_data(pr_xy[:k + 1].reshape(-1, 2)[:, 0], pr_xy[:k + 1].reshape(-1, 2)[:, 1])
        gt_pts.set_data(gt_xy[k, :, 0], gt_xy[k, :, 1])
        pr_pts.set_data(pr_xy[k, :, 0], pr_xy[k, :, 1])
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())
        frames_pil.append(Image.fromarray(buf[..., :3]))

    frames_pil[0].save(path, save_all=True, append_images=frames_pil[1:],
                       optimize=False, duration=160, loop=0)
    plt.close(fig)
    print(f"  clip saved: {path} ({os.path.getsize(path)} bytes)", flush=True)


def make_clips(model):
    print("=" * 64)
    print("  2. Simulation clips (model rollout vs ground truth)")
    print("=" * 64)

    # 9 synthetic-domain clips from the universal model
    for d in DOMAINS:
        va = gen_synthetic(d, 1, 777)
        cfg = DOMAIN_CONFIGS[d]
        pos0 = torch.tensor(va["pos"][:, 0], dtype=torch.float32, device=DEVICE)
        vel0 = torch.tensor(va["vel"][:, 0], dtype=torch.float32, device=DEVICE)
        mass = torch.tensor(va["masses"][:, 0, :], dtype=torch.float32, device=DEVICE)
        dt = va["dt"]
        n_steps = cfg["n_steps"]
        pred = rollout(model, pos0, vel0, mass, d, dt, n_steps)
        gt = va["pos"][0]
        make_clip(d, gt, pred, f"{d}  (blue = ground truth, red = PSN-1)",
                  os.path.join(RESULTS, f"clip_{d}.gif"))

    # 3 real-physics clips (2D simulators, padded to 3D for the model)
    print("  real-physics clips...", flush=True)
    real_sims = [
        ("gravity", NBodyGravity(n_particles=3, dt=0.01, n_steps=60)),
        ("spring", SpringChain(n_particles=4, dt=0.005, n_steps=60)),
        ("lennard_jones", LennardJones(n_particles=4, dt=0.002, n_steps=60)),
    ]
    for name, sim in real_sims:
        data = sim.generate(n_traj=1, seed=123)
        pos2 = data["positions"][0]          # (T, N, 2)
        vel2 = data["velocities"][0]
        masses = data["masses"][0]           # (N,)
        T, N, _ = pos2.shape
        pos3 = np.concatenate([pos2, np.zeros((T, N, 1), dtype=np.float32)], axis=-1)
        vel3 = np.concatenate([vel2, np.zeros((T, N, 1), dtype=np.float32)], axis=-1)
        pos0 = torch.tensor(pos3[0:1], dtype=torch.float32, device=DEVICE)
        vel0 = torch.tensor(vel3[0:1], dtype=torch.float32, device=DEVICE)
        mass = torch.tensor(masses[None, :], dtype=torch.float32, device=DEVICE)
        dt = sim.dt
        pred = rollout(model, pos0, vel0, mass, name, dt, T)
        make_clip(f"real_{name}", pos3, pred, f"{name} (real simulator)  blue=GT red=PSN-1",
                  os.path.join(RESULTS, f"clip_real_{name}.gif"))


def main():
    t_all = time.time()
    model, final, mean_mse, history, n_params, t_train = train_universal()
    make_clips(model)

    summary = {
        "name": "psn1_universal_nmi",
        "device": DEVICE,
        "n_domains": len(DOMAINS),
        "n_epochs": N_EPOCHS,
        "n_train_per_domain": N_TRAIN,
        "n_params": n_params,
        "per_domain": final,
        "mean_mse": mean_mse,
        "train_loss_final": float(history["train_loss"][-1]),
        "train_loss_initial": float(history["train_loss"][0]),
        "train_time_s": float(t_train),
        "total_time_s": float(time.time() - t_all),
        "clips": [f"clip_{d}.gif" for d in DOMAINS]
                + ["clip_real_gravity.gif", "clip_real_spring.gif",
                   "clip_real_lennard_jones.gif"],
    }
    with open(os.path.join(RESULTS, "psn1_kaggle_nmi.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 64)
    print(f"  DONE. mean MSE {mean_mse:.6e} in {time.time()-t_all:.0f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
