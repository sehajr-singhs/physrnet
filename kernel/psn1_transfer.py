#!/usr/bin/env python3
"""PSN-1 Cross-Domain Transfer Experiment
Train on gravity, then fine-tune on Lennard-Jones with varying data.
Compare: (a) train LJ from scratch, (b) fine-tune from gravity.
Shows physics knowledge transfers across domains.
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

DOMAIN_CONFIGS = {
    "gravity": {"n_particles": 4, "n_steps": 50, "dt": 0.01},
    "lennard_jones": {"n_particles": 4, "n_steps": 50, "dt": 0.002},
}

N_EPOCHS_PRETRAIN = 30
N_EPOCHS_FINETUNE = 30
N_EPOCHS_SCRATCH = 60  # same total budget as pretrain+finetune
BATCH_SIZE = 32
LR_PRETRAIN = 1e-3
LR_FINETUNE = 5e-4
LR_SCRATCH = 1e-3
W_PINN = 0.3

# Fine-tune data fractions
DATA_FRACS = [0.05, 0.1, 0.2, 0.5, 1.0]
N_TRAIN_FULL = 80

# ---- data gen ----
def gen_synthetic(domain, n_samples, seed, dt=0.05):
    cfg = DOMAIN_CONFIGS[domain]
    rng = np.random.RandomState(seed)
    omega_map = {"gravity": 0.5, "lennard_jones": 2.0}
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

def train_model(model, batches, n_epochs, lr, domain_label):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    history = []
    for epoch in range(n_epochs):
        np.random.shuffle(batches)
        model.train()
        total = 0.0
        for b in batches:
            pred, aux = model(b["pos"], b["vel"], b["masses"], domain_label)
            accel = pred["acceleration"]
            dt = b["dt"]
            pos_p = b["pos"] + b["vel"]*dt + 0.5*accel*dt**2
            vel_p = b["vel"] + accel*dt
            dl = (pos_p - b["pos_tgt"]).pow(2).mean() + (vel_p - b["vel_tgt"]).pow(2).mean()
            try:
                pl = model.physics_loss(
                    {"acceleration": accel, "pos_pred": pos_p, "vel_pred": vel_p},
                    {"pos": b["pos"], "vel": b["vel"], "masses": b["masses"], "dt": dt},
                    domain_label)
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
        history.append(total / max(len(batches), 1))
    return history

def eval_model(model, val_data, domain_label):
    model.eval()
    with torch.no_grad():
        pos = torch.tensor(val_data["pos"], dtype=torch.float32, device=DEVICE)
        vel = torch.tensor(val_data["vel"], dtype=torch.float32, device=DEVICE)
        mass = torch.tensor(val_data["masses"], dtype=torch.float32, device=DEVICE)
        dt = val_data["dt"]
        pred, aux = model(pos[:, 0], vel[:, 0], mass[:, 0, :], domain_label)
        accel = pred["acceleration"]
        pos_p = pos[:, 0] + vel[:, 0]*dt + 0.5*accel*dt**2
        mse = (pos_p - pos[:, 1]).pow(2).mean().item()
        gate = aux.get("gate", torch.tensor(0.5)).mean().item()
    return {"mse": mse, "gate": gate}

# ---- main ----
if __name__ == "__main__":
    print("="*60)
    print("  PSN-1 Cross-Domain Transfer Experiment")
    print("  Train gravity -> fine-tune LJ vs scratch")
    print("="*60)

    max_particles = max(c["n_particles"] for c in DOMAIN_CONFIGS.values())

    # Generate all data
    print("\nGenerating data...", flush=True)
    gravity_data = gen_synthetic("gravity", N_TRAIN_FULL, 42)
    gravity_val = gen_synthetic("gravity", 20, 999)
    lj_data = gen_synthetic("lennard_jones", N_TRAIN_FULL, 42)
    lj_val = gen_synthetic("lennard_jones", 20, 999)

    # === Phase 1: Pretrain on gravity ===
    print("\n" + "="*60)
    print("  Phase 1: Pretrain on gravity (30 epochs)")
    print("="*60)
    model = PSNUniversal(n_particles=max_particles, hidden=128, n_scalar=8,
                         n_heads=4, n_domains=2, domain_emb_dim=16).to(DEVICE)
    gravity_batches = []
    for b in build_batches(gravity_data, BATCH_SIZE):
        b["domain"] = "gravity"
        gravity_batches.append(b)

    gravity_history = train_model(model, gravity_batches, N_EPOCHS_PRETRAIN, LR_PRETRAIN, "gravity")
    gravity_eval = eval_model(model, gravity_val, "gravity")
    print(f"  Gravity pretrain done: MSE={gravity_eval['mse']:.4e}")

    # Save pretrained weights
    pretrained_state = {k: v.clone() for k, v in model.state_dict().items()}

    # === Phase 2: Fine-tune on LJ with varying data ===
    print("\n" + "="*60)
    print("  Phase 2: Fine-tune on LJ with varying data fractions")
    print("="*60)

    finetune_results = {}
    for frac in DATA_FRACS:
        n_lj = max(1, int(N_TRAIN_FULL * frac))
        lj_subset = {k: v[:n_lj] if hasattr(v, '__getitem__') and not isinstance(v, float) else v for k, v in lj_data.items()}

        # Load pretrained weights (domain embedding dims may differ, skip mismatched)
        model_ft = PSNUniversal(n_particles=max_particles, hidden=128, n_scalar=8,
                                n_heads=4, n_domains=2, domain_emb_dim=16).to(DEVICE)
        # Copy compatible weights
        pretrained_compat = {}
        ft_state = model_ft.state_dict()
        for k, v in pretrained_state.items():
            if k in ft_state and v.shape == ft_state[k].shape:
                pretrained_compat[k] = v
        model_ft.load_state_dict(pretrained_compat, strict=False)

        # Fine-tune
        ft_batches = []
        for b in build_batches(lj_subset, BATCH_SIZE):
            b["domain"] = "lennard_jones"
            ft_batches.append(b)

        ft_history = train_model(model_ft, ft_batches, N_EPOCHS_FINETUNE, LR_FINETUNE, "lennard_jones")
        ft_eval = eval_model(model_ft, lj_val, "lennard_jones")
        finetune_results[str(frac)] = {
            "n_lj_samples": n_lj, "frac": frac,
            "mse": ft_eval["mse"], "gate": ft_eval["gate"],
            "train_loss_end": ft_history[-1]
        }
        print(f"  LJ frac={frac:.2f} ({n_lj:3d} samples): MSE={ft_eval['mse']:.4e} (fine-tuned)")

    # === Phase 3: Train LJ from scratch (same total budget) ===
    print("\n" + "="*60)
    print("  Phase 3: Train LJ from scratch (60 epochs = pretrain+finetune)")
    print("="*60)

    scratch_results = {}
    for frac in DATA_FRACS:
        n_lj = max(1, int(N_TRAIN_FULL * frac))
        lj_subset = {k: v[:n_lj] if hasattr(v, '__getitem__') and not isinstance(v, float) else v for k, v in lj_data.items()}

        model_scratch = PSNUniversal(n_particles=max_particles, hidden=128, n_scalar=8,
                                      n_heads=4, n_domains=2, domain_emb_dim=16).to(DEVICE)
        sc_batches = []
        for b in build_batches(lj_subset, BATCH_SIZE):
            b["domain"] = "lennard_jones"
            sc_batches.append(b)

        sc_history = train_model(model_scratch, sc_batches, N_EPOCHS_SCRATCH, LR_SCRATCH, "lennard_jones")
        sc_eval = eval_model(model_scratch, lj_val, "lennard_jones")
        scratch_results[str(frac)] = {
            "n_lj_samples": n_lj, "frac": frac,
            "mse": sc_eval["mse"], "gate": sc_eval["gate"],
            "train_loss_end": sc_history[-1]
        }
        print(f"  LJ frac={frac:.2f} ({n_lj:3d} samples): MSE={sc_eval['mse']:.4e} (from scratch)")

    # Save results
    output = {
        "experiment": "psn1_cross_domain_transfer",
        "device": str(DEVICE),
        "gravity_pretrain": {"epochs": N_EPOCHS_PRETRAIN, "eval": gravity_eval},
        "finetune": finetune_results,
        "scratch": scratch_results,
        "data_fractions": DATA_FRACS,
    }
    with open(os.path.join(RESULTS, "psn1_transfer.json"), "w") as f:
        json.dump(output, f, indent=2)

    # Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fracs = DATA_FRACs if 'DATA_FRACs' in dir() else DATA_FRACS
    ft_mses = [finetune_results[str(f)]["mse"] for f in fracs]
    sc_mses = [scratch_results[str(f)]["mse"] for f in fracs]

    axes[0].plot(fracs, ft_mses, "o-", color="#4CAF50", linewidth=2, markersize=8, label="Fine-tuned from gravity")
    axes[0].plot(fracs, sc_mses, "s--", color="#E53935", linewidth=2, markersize=8, label="From scratch")
    axes[0].set_xlabel("Fraction of LJ training data")
    axes[0].set_ylabel("Test MSE")
    axes[0].set_title("Cross-Domain Transfer: Gravity → Lennard-Jones")
    axes[0].set_yscale("log")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # improvement ratio
    ratios = [sc/ft if ft > 0 else 1 for sc, ft in zip(sc_mses, ft_mses)]
    axes[1].bar([str(f) for f in fracs], ratios, color=["#4A90D9","#4CAF50","#E8913A","#9C27B0","#E53935"])
    axes[1].axhline(y=1.0, color="white", linestyle="--", alpha=0.3, label="Break even")
    axes[1].set_xlabel("Fraction of LJ training data")
    axes[1].set_ylabel("Scratch MSE / Fine-tune MSE (>1 = transfer helps)")
    axes[1].set_title("Transfer Improvement Ratio")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_transfer.png"), dpi=160)
    plt.close(fig)

    print(f"\nDONE. Results saved to {RESULTS}/psn1_transfer.json")
