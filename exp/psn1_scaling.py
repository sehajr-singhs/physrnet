#!/usr/bin/env python3
"""PSN-1 scaling law experiment.

Trains PSN-1 at 4 model sizes (tiny/small/medium/full) on all 9 domains,
reports MSE vs parameter count for a scaling law curve.
"""
import json, os, sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exp.psn1_kaggle_nmi import (
    DOMAINS, DOMAIN_CONFIGS, gen_synthetic, build_batches,
    N_EPOCHS, N_TRAIN, N_VAL, BATCH_SIZE, LR, W_PINN
)
from physrnet.model_universal import PSNUniversal

DEVICE = "cpu"
N_EPOCHS = 25
N_TRAIN = 40
RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)

# Scale configs: (hidden, n_scalar, n_heads) -> gives different param counts
SCALE_CONFIGS = {
    "tiny":   {"hidden": 16, "n_scalar": 4, "n_heads": 2},
    "small":  {"hidden": 32, "n_scalar": 6, "n_heads": 4},
    "medium": {"hidden": 48, "n_scalar": 8, "n_heads": 4},
    "full":   {"hidden": 64, "n_scalar": 8, "n_heads": 4},
}


def train_scale(name, cfg):
    torch.manual_seed(42)
    np.random.seed(42)

    model = PSNUniversal(
        n_particles=max(c["n_particles"] for c in DOMAIN_CONFIGS.values()),
        hidden=cfg["hidden"], n_scalar=cfg["n_scalar"], n_heads=cfg["n_heads"],
        n_domains=len(DOMAINS), domain_emb_dim=16,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())

    batches = []
    val_data = {}
    for d in DOMAINS:
        tr = gen_synthetic(d, N_TRAIN, 42)
        va = gen_synthetic(d, N_VAL, 999)
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
    print(f"PSN-1 scaling law: {len(SCALE_CONFIGS)} sizes × {len(DOMAINS)} domains")
    t0 = time.time()

    all_results = {}
    for name, cfg in SCALE_CONFIGS.items():
        print(f"\n--- {name} (hidden={cfg['hidden']}) ---", flush=True)
        t1 = time.time()
        res, n_params = train_scale(name, cfg)
        mean_mse = np.mean([r["mse"] for r in res.values()])
        all_results[name] = {
            "config": cfg,
            "n_params": n_params,
            "per_domain": res,
            "mean_mse": float(mean_mse),
        }
        print(f"  {n_params:,} params, mean MSE={mean_mse:.4e}, {time.time()-t1:.0f}s", flush=True)

    # Summary
    sizes = list(all_results.keys())
    params = [all_results[s]["n_params"] for s in sizes]
    mses = [all_results[s]["mean_mse"] for s in sizes]

    # Fit power law: MSE = a * N^b
    log_params = np.log(params)
    log_mses = np.log(mses)
    b, a = np.polyfit(log_params, log_mses, 1)
    a = np.exp(a)

    summary = {
        "sizes": sizes,
        "per_size": {s: {
            "n_params": all_results[s]["n_params"],
            "mean_mse": all_results[s]["mean_mse"],
            "per_domain": all_results[s]["per_domain"],
        } for s in sizes},
        "scaling_law": {"exponent": float(b), "coefficient": float(a)},
        "total_time_s": time.time() - t0,
    }

    out_path = os.path.join(RESULTS, "psn1_scaling.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("SCALING LAW RESULTS:")
    print(f"{'='*60}")
    for s in sizes:
        print(f"  {s:8s}  {all_results[s]['n_params']:>8,} params  MSE={all_results[s]['mean_mse']:.4e}")
    print(f"  Power law: MSE = {a:.2e} × N^{b:.2f}")
    print(f"  Total: {time.time()-t0:.0f}s")

    # Figure
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(params, mses, "o-", color="#2563eb", linewidth=2, markersize=8)
    ax.set_xlabel("Parameters", fontsize=12)
    ax.set_ylabel("Mean MSE (9 domains)", fontsize=12)
    ax.set_title(f"PSN-1 Scaling Law (exponent = {b:.2f})")
    ax.grid(True, alpha=0.3)
    for i, s in enumerate(sizes):
        ax.annotate(s, (params[i], mses[i]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_scaling.png"), dpi=160)
    plt.close(fig)
    print(f"  Figure: {RESULTS}/fig_scaling.png")


if __name__ == "__main__":
    main()
