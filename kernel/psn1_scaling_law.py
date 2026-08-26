#!/usr/bin/env python3
"""PSN-1 Scaling Law Experiment
Train at 4 model sizes on 9 physics domains.
Reveals power-law relationship between parameters and performance.
Runs on Kaggle T4 GPU.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from physrnet.model_universal import PSNUniversal

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, flush=True)

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

# 4 model sizes
SCALES = {
    "1K":   {"hidden": 16,  "n_scalar": 2, "n_heads": 2},
    "10K":  {"hidden": 32,  "n_scalar": 4, "n_heads": 2},
    "50K":  {"hidden": 64,  "n_scalar": 8, "n_heads": 4},
    "157K": {"hidden": 128, "n_scalar": 8, "n_heads": 4},
}

N_EPOCHS = 40
N_TRAIN = 60
N_VAL = 10
BATCH_SIZE = 32
LR = 1e-3
W_PINN = 0.3

# ---- data gen (reused from psn1_kaggle_nmi.py) ----
def gen_synthetic(domain, n_samples, seed, dt=0.05):
    cfg = DOMAIN_CONFIGS[domain]
    rng = np.random.RandomState(seed)
    omega_map = {"gravity": 0.5, "spring": 1.5, "lennard_jones": 2.0,
                 "fluid": 0.3, "electromagnetism": 0.8, "quantum": 1.2,
                 "heat": 0.1, "relativistic": 0.6, "thermo_ideal": 0.2}
    omega = omega_map.get(domain, 1.0)
    damping = 0.05
    N = cfg["n_particles"]; T = cfg["n_steps"]; D = 3
    pos = np.zeros((n_samples, T, N, D), dtype=np.float32)
    vel = np.zeros((n_samples, T, N, D), dtype=np.float32)
    for i in range(n_samples):
        amp = rng.uniform(0.5, 2.0, (N, D))
        phase = rng.uniform(0, 2*np.pi, (N, D))
        for t in range(T):
            tv = t * dt; env = np.exp(-damping * omega * tv)
            pos[i, t] = amp * env * np.cos(omega*tv + phase)
            vel[i, t] = -amp * env * omega * np.sin(omega*tv + phase)
    masses = np.ones((n_samples, T, N), dtype=np.float32)
    return {"pos": pos, "vel": vel, "masses": masses, "dt": dt}

def build_batches(data, batch_size):
    pos = torch.tensor(data["pos"], dtype=torch.float32, device=DEVICE)
    vel = torch.tensor(data["vel"], dtype=torch.float32, device=DEVICE)
    mass = torch.tensor(data["masses"], dtype=torch.float32, device=DEVICE)
    S, T, N, D = pos.shape
    in_pos = pos[:, :-1].reshape(-1, N, D)
    in_vel = vel[:, :-1].reshape(-1, N, D)
    m = mass[:, :T-1].reshape(-1, N)
    tgt_p = pos[:, 1:].reshape(-1, N, D)
    tgt_v = vel[:, 1:].reshape(-1, N, D)
    out = []
    for start in range(0, in_pos.shape[0], batch_size):
        end = min(start + batch_size, in_pos.shape[0])
        out.append({"pos": in_pos[start:end], "vel": in_vel[start:end],
                     "masses": m[start:end], "pos_tgt": tgt_p[start:end],
                     "vel_tgt": tgt_v[start:end], "dt": data["dt"]})
    return out

# ---- train one scale ----
def train_scale(scale_name, scale_cfg):
    print(f"\n{'='*60}")
    print(f"  Scale: {scale_name}")
    print(f"  hidden={scale_cfg['hidden']}, n_scalar={scale_cfg['n_scalar']}, n_heads={scale_cfg['n_heads']}")
    print(f"{'='*60}")

    model = PSNUniversal(
        n_particles=max(c["n_particles"] for c in DOMAIN_CONFIGS.values()),
        hidden=scale_cfg["hidden"],
        n_scalar=scale_cfg["n_scalar"],
        n_heads=scale_cfg["n_heads"],
        n_domains=len(DOMAINS),
        domain_emb_dim=16,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")

    # generate data
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

    t0 = time.time()
    for epoch in range(N_EPOCHS):
        te = time.time()
        np.random.shuffle(batches)
        model.train()
        total = 0.0
        for b in batches:
            pred, aux = model(b["pos"], b["vel"], b["masses"], b["domain"])
            accel = pred["acceleration"]
            dt = b["dt"]
            pos_p = b["pos"] + b["vel"]*dt + 0.5*accel*dt**2
            vel_p = b["vel"] + accel*dt
            dl = (pos_p - b["pos_tgt"]).pow(2).mean() + (vel_p - b["vel_tgt"]).pow(2).mean()
            try:
                pl = model.physics_loss(
                    {"acceleration": accel, "pos_pred": pos_p, "vel_pred": vel_p},
                    {"pos": b["pos"], "vel": b["vel"], "masses": b["masses"], "dt": dt},
                    b["domain"])
                if not isinstance(pl, torch.Tensor): pl = torch.tensor(0.0, device=DEVICE)
                if pl.dim() > 0: pl = pl.mean()
            except: pl = torch.tensor(0.0, device=DEVICE)
            loss = dl + W_PINN * pl
            gate = aux.get("gate", torch.ones(1, device=DEVICE))
            loss = loss + 0.01 * (gate * (1 - gate)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        if (epoch+1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}: loss={total/len(batches):.4f} [{time.time()-te:.0f}s]", flush=True)

    train_time = time.time() - t0

    # eval
    results = {"params": n_params, "per_domain": {}, "train_time_s": train_time}
    model.eval()
    with torch.no_grad():
        for d in DOMAINS:
            va = val_data[d]
            pos = torch.tensor(va["pos"], dtype=torch.float32, device=DEVICE)
            vel = torch.tensor(va["vel"], dtype=torch.float32, device=DEVICE)
            mass = torch.tensor(va["masses"], dtype=torch.float32, device=DEVICE)
            dt = va["dt"]
            pred, aux = model(pos[:, 0], vel[:, 0], mass[:, 0, :], d)
            accel = pred["acceleration"]
            pos_p = pos[:, 0] + vel[:, 0]*dt + 0.5*accel*dt**2
            mse = (pos_p - pos[:, 1]).pow(2).mean().item()
            gate = aux.get("gate", torch.tensor(0.5)).mean().item()
            results["per_domain"][d] = {"mse": mse, "gate": gate}
            print(f"    {d:20s}: MSE={mse:.4e}  gate={gate:.4e}")

    mean_mse = float(np.mean([v["mse"] for v in results["per_domain"].values()]))
    results["mean_mse"] = mean_mse
    print(f"  Mean MSE: {mean_mse:.4e} | Time: {train_time:.0f}s")
    return results

# ---- main ----
if __name__ == "__main__":
    print("="*60)
    print("  PSN-1 Scaling Law Experiment")
    print("  4 scales x 9 domains x 40 epochs")
    print("="*60)

    all_results = {}
    for name, cfg in SCALES.items():
        all_results[name] = train_scale(name, cfg)

    # power law fit
    params = np.array([v["params"] for v in all_results.values()])
    mses = np.array([v["mean_mse"] for v in all_results.values()])
    log_p = np.log10(params); log_m = np.log10(mses)
    alpha, log_A = np.polyfit(log_p, log_m, 1)

    print(f"\n{'='*60}")
    print("  SCALING LAW SUMMARY")
    print(f"{'='*60}")
    for k, v in all_results.items():
        print(f"  {k:8s}: {v['params']:>8,} params, MSE={v['mean_mse']:.4e}")
    print(f"\n  Power law: MSE = 10^{log_A:.2f} * N^({alpha:.2f})")
    print(f"  Scaling exponent: {-alpha:.2f} (MSE ∝ N^(-{-alpha:.2f}))")

    output = {
        "experiment": "psn1_scaling_law",
        "device": str(DEVICE),
        "n_epochs": N_EPOCHS,
        "scales": {k: {"params": v["params"], "mean_mse": v["mean_mse"],
                        "per_domain": v["per_domain"], "train_time_s": v["train_time_s"]}
                   for k, v in all_results.items()},
        "power_law": {"alpha": float(-alpha), "log_A": float(log_A)},
    }
    with open(os.path.join(RESULTS, "psn1_scaling_law.json"), "w") as f:
        json.dump(output, f, indent=2)

    # scaling figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    names = list(SCALES.keys())
    p_arr = [all_results[n]["params"] for n in names]
    m_arr = [all_results[n]["mean_mse"] for n in names]
    # left: bar chart
    axes[0].bar(names, [-np.log10(m) for m in m_arr], color=["#4A90D9","#4CAF50","#E8913A","#E53935"])
    axes[0].set_ylabel("-log10(MSE)")
    axes[0].set_title(f"PSN-1 Scaling: alpha={-alpha:.2f}")
    for i, (n, m) in enumerate(zip(names, m_arr)):
        axes[0].text(i, -np.log10(m)+0.1, f"{m:.1e}", ha="center", fontsize=9)
    # right: log-log
    axes[1].loglog(p_arr, m_arr, "o-", color="#4A90D9", linewidth=2, markersize=8)
    fit_p = np.logspace(np.log10(p_arr[0])*0.8, np.log10(p_arr[-1])*1.1, 50)
    fit_m = 10**log_A * fit_p**alpha
    axes[1].loglog(fit_p, fit_m, "--", color="#E53935", alpha=0.5, label=f"α={-alpha:.2f}")
    axes[1].set_xlabel("Parameters"); axes[1].set_ylabel("Mean Test MSE")
    axes[1].set_title("Log-Log Scaling")
    axes[1].legend(); axes[1].grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_scaling_law.png"), dpi=160)
    plt.close(fig)

    print(f"\nDONE. Results saved to {RESULTS}/psn1_scaling_law.json")
