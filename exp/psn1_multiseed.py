#!/usr/bin/env python3
"""PSN-1 multi-seed experiment for error bars.

Trains PSN-1 with 3 different random seeds on all 9 domains,
reports mean ± std for each domain's MSE and gate value.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

# Import the exact same training code from the Kaggle script
from exp.psn1_kaggle_nmi import (
    DOMAINS, DOMAIN_CONFIGS, gen_synthetic, build_batches,
    N_EPOCHS, N_TRAIN, N_VAL, BATCH_SIZE, LR, W_PINN
)
from physrnet.model_universal import PSNUniversal

DEVICE = "cpu"
N_SEEDS = 3
N_EPOCHS = 25
N_TRAIN = 40
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

def train_one_seed(seed):
    """Train PSN-1 from scratch with a given seed, return per-domain results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = PSNUniversal(
        n_particles=max(c["n_particles"] for c in DOMAIN_CONFIGS.values()),
        hidden=64, n_scalar=8, n_heads=4,
        n_domains=len(DOMAINS), domain_emb_dim=16,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    # Generate data
    batches = []
    val_data = {}
    for d in DOMAINS:
        tr = gen_synthetic(d, N_TRAIN, seed)
        va = gen_synthetic(d, N_VAL, seed + 1000)
        val_data[d] = va
        for b in build_batches(tr, BATCH_SIZE):
            b["domain"] = d
            batches.append(b)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)

    for epoch in range(N_EPOCHS):
        np.random.shuffle(batches)
        model.train()
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
            except:
                pl = torch.tensor(0.0, device=DEVICE)
            loss = dl + W_PINN * pl
            gate = aux.get("gate", torch.ones(1, device=DEVICE))
            loss = loss + 0.01 * (gate * (1 - gate)).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    # Eval
    results = {}
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
            results[d] = {"mse": mse, "gate": gate}
    return results, n_params


def main():
    print(f"PSN-1 multi-seed: {N_SEEDS} seeds × {len(DOMAINS)} domains, {N_EPOCHS} epochs each")
    t0 = time.time()

    all_results = []
    for seed in range(N_SEEDS):
        print(f"\n--- Seed {seed} ---", flush=True)
        t1 = time.time()
        res, n_params = train_one_seed(seed)
        all_results.append(res)
        print(f"  Done in {time.time()-t1:.0f}s", flush=True)

    # Aggregate: mean ± std per domain
    agg = {}
    for d in DOMAINS:
        mses = [r[d]["mse"] for r in all_results]
        gates = [r[d]["gate"] for r in all_results]
        agg[d] = {
            "mse_mean": float(np.mean(mses)),
            "mse_std": float(np.std(mses)),
            "mse_min": float(np.min(mses)),
            "mse_max": float(np.max(mses)),
            "gate_mean": float(np.mean(gates)),
            "gate_std": float(np.std(gates)),
            "per_seed_mse": mses,
            "per_seed_gate": gates,
        }

    mean_mse = float(np.mean([agg[d]["mse_mean"] for d in DOMAINS]))
    mean_mse_std = float(np.std([agg[d]["mse_mean"] for d in DOMAINS]))

    summary = {
        "n_seeds": N_SEEDS,
        "n_domains": len(DOMAINS),
        "n_epochs": N_EPOCHS,
        "n_train_per_domain": N_TRAIN,
        "n_params": n_params,
        "device": DEVICE,
        "per_domain": agg,
        "mean_mse": mean_mse,
        "mean_mse_std": mean_mse_std,
        "total_time_s": time.time() - t0,
    }

    out_path = os.path.join(RESULTS, "psn1_multiseed.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS (mean ± std over {N_SEEDS} seeds):")
    print(f"{'='*60}")
    for d in DOMAINS:
        a = agg[d]
        print(f"  {d:20s}  MSE={a['mse_mean']:.4e} ± {a['mse_std']:.4e}  gate={a['gate_mean']:.6f} ± {a['gate_std']:.6f}")
    print(f"  {'MEAN':20s}  MSE={mean_mse:.4e} ± {mean_mse_std:.4e}")
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
