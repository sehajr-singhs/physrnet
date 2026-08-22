"""Generate paper figures from committed result JSONs.

Usage:
    python benchmarks/make_figures.py --results results --figs figs
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

BLUE = "#226999"
GREEN = "#2e6b4f"
RED = "#b03a2e"
GOLD = "#b08a2e"
PURPLE = "#7a5aa8"
GRAY = "#8c8e90"


def load(name):
    with open(name) as f:
        return json.load(f)


def fig_training_curves(results_dir, figs_dir):
    """Fig 1: training loss curves for three systems."""
    systems = [
        ("nbody_gravity", "N-body Gravity", BLUE),
        ("spring_chain", "Spring Chain", GREEN),
        ("lennard_jones", "Lennard-Jones", GOLD),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (key, label, color) in zip(axes, systems):
        try:
            h = load(os.path.join(results_dir, key, "history.json"))
        except FileNotFoundError:
            h = load(os.path.join(results_dir, "history.json"))
        epochs = np.arange(1, len(h["train_mse"]) + 1)
        ax.semilogy(epochs, h["train_mse"], color=color, label="Train", lw=1.5)
        val_epochs = np.arange(1, len(h["val_mse"]) + 1) * 5
        val_epochs = val_epochs[: len(h["val_mse"])]
        ax.semilogy(val_epochs[:len(h["val_mse"])], h["val_mse"], color=color, ls="--", label="Val", lw=1.5)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend(frameon=False)
        ax.set_xlim(0, max(epochs))
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "fig1_training.png"))
    plt.close(fig)
    print(f"  fig1_training.png")


def fig_equivariance(results_dir, figs_dir):
    """Fig 2: rotation equivariance error across systems and ablations."""
    experiments = [
        ("nbody_gravity", "Gravity"),
        ("spring_chain", "Spring"),
        ("lennard_jones", "LJ"),
        ("reasoning_only_spring", "R-only Spring"),
        ("reasoning_only_gravity", "R-only Gravity"),
        ("ablation_no_pinn", "No PINN"),
    ]
    names, equiv_errs = [], []
    for key, label in experiments:
        try:
            r = load(os.path.join(results_dir, f"{key}.json"))
            names.append(label)
            equiv_errs.append(r.get("test_equivariance_err", 1e-10))
        except FileNotFoundError:
            pass
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.array(equiv_errs)
    colors = [BLUE, GREEN, GOLD, GRAY, RED, PURPLE][: len(names)]
    bars = ax.bar(range(len(names)), y, color=colors, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Rotation equivariance error")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-9)
    ax.axhline(1e-10, color=GRAY, ls=":", lw=0.8, label="machine epsilon")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Exactly E(2)-equivariant by construction")
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "fig2_equivariance.png"))
    plt.close(fig)
    print(f"  fig2_equivariance.png")


def fig_comparison(results_dir, figs_dir):
    """Fig 3: test MSE and energy drift across systems + ablation."""
    experiments = [
        ("nbody_gravity", "Gravity"),
        ("spring_chain", "Spring"),
        ("lennard_jones", "LJ"),
        ("ablation_no_pinn", "No PINN"),
        ("reasoning_only_spring", "R-only Spring"),
        ("reasoning_only_gravity", "R-only Gravity"),
    ]
    names, mses, drifts, equivs = [], [], [], []
    for key, label in experiments:
        try:
            r = load(os.path.join(results_dir, f"{key}.json"))
            names.append(label)
            mses.append(r.get("test_mse", np.nan))
            drifts.append(r.get("test_energy_drift", np.nan))
            equivs.append(r.get("test_equivariance_err", np.nan))
        except FileNotFoundError:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = [BLUE, GREEN, GOLD, PURPLE, GRAY, RED][: len(names)]

    ax = axes[0]
    ax.bar(range(len(names)), mses, color=colors, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Test MSE")
    ax.set_title("Force-law learning")

    ax = axes[1]
    ax.bar(range(len(names)), drifts, color=colors, edgecolor="white", lw=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Energy drift (50-step rollout)")
    ax.set_title("Conservation: with vs without PINN")

    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "fig3_comparison.png"))
    plt.close(fig)
    print(f"  fig3_comparison.png")


def fig_generalization(results_dir, figs_dir):
    """Fig 4: generalization — test vs gen MSE."""
    experiments = [
        ("nbody_gravity", "Gravity"),
        ("spring_chain", "Spring"),
        ("lennard_jones", "LJ"),
    ]
    names, test_mses, gen_mses = [], [], []
    for key, label in experiments:
        try:
            r = load(os.path.join(results_dir, f"{key}.json"))
            names.append(label)
            test_mses.append(r.get("test_mse", np.nan))
            gen_mses.append(r.get("gen_mse", np.nan))
        except FileNotFoundError:
            pass

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = np.arange(len(names))
    w = 0.35
    colors = [BLUE, GREEN, GOLD]
    ax.bar(x - w / 2, test_mses, w, label="Test (seen masses)", color=colors, edgecolor="white")
    ax.bar(x + w / 2, gen_mses, w, label="Generalization (unseen)", color=colors, alpha=0.55, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("MSE")
    ax.set_title("Generalization to unseen masses and initial conditions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "fig4_generalization.png"))
    plt.close(fig)
    print(f"  fig4_generalization.png")


def fig_spring_interpret(results_dir, figs_dir):
    """Fig 5: spring chain interaction graph — learned edge weights vs true force."""
    try:
        r = load(os.path.join(results_dir, "spring_chain.json"))
    except FileNotFoundError:
        return
    fig, ax = plt.subplots(figsize=(5, 3.5))
    topo = r.get("test_graph_topology_recall")
    strength = r.get("test_graph_strength_corr")
    adj = r.get("test_adjacency_auc")
    labels = []
    vals = []
    if topo is not None:
        labels.append("Topology\nrecall")
        vals.append(topo)
    if adj is not None:
        labels.append("Adjacency\nAUC")
        vals.append(adj)
    if labels:
        ax.bar(range(len(labels)), vals, color=GREEN, edgecolor="white", lw=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Interaction-graph recovery (spring chain)")
        ax.axhline(0.5, color=GRAY, ls=":", lw=0.8, label="chance")
        ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 1)
    else:
        ax.text(0.5, 0.5, "No spring topology data", ha="center", va="center",
                transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "fig5_interpret.png"))
    plt.close(fig)
    print(f"  fig5_interpret.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--figs", default="figs")
    args = ap.parse_args()
    os.makedirs(args.figs, exist_ok=True)
    print("Generating figures...")
    fig_training_curves(args.results, args.figs)
    fig_equivariance(args.results, args.figs)
    fig_comparison(args.results, args.figs)
    fig_generalization(args.results, args.figs)
    fig_spring_interpret(args.results, args.figs)
    print(f"All figures saved to {args.figs}/")


if __name__ == "__main__":
    main()
