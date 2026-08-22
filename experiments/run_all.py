"""Run the full PhysRNet experiment suite.

Trains and evaluates PhysRNet on three conservative particle systems
(N-body gravity, spring chain, Lennard-Jones fluid) and runs the
physics-informed (PINN) ablation. Saves results to results/*.json.

Usage:
    python experiments/run_all.py [--epochs 60] [--out results] [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physrnet.datasets import NBodyGravity, SpringChain, LennardJones
from physrnet.model import PhysRNet
from physrnet.training import train_physrnet, evaluate_physrnet


def run_experiment(name, dataset, model, out_dir, n_train, n_val, n_test,
                   n_epochs, dt, device, w_physics=0.5) -> dict:
    print(f"\n{'=' * 64}\n  {name}\n{'=' * 64}")
    t0 = time.time()
    train_data = dataset.generate(n_traj=n_train, seed=42)
    val_data = dataset.generate(n_traj=n_val, seed=43)
    test_data = dataset.generate(n_traj=n_test, seed=44)
    print(f"  data generated ({time.time()-t0:.1f}s)")

    train_physrnet(model, train_data, val_data, n_epochs=n_epochs,
                   w_physics=w_physics, device=device, out_dir=out_dir, dt=dt,
                   batch_size=512)

    test_metrics = evaluate_physrnet(model, test_data, device, dt=dt, n_eval=400)
    gen_data = dataset.generate(n_traj=n_test, seed=99)
    gen_metrics = evaluate_physrnet(model, gen_data, device, dt=dt, n_eval=400)

    results = {
        "name": name,
        "physics_type": model.physics.physics_type,
        "n_train": n_train, "n_val": n_val, "n_test": n_test,
        "n_epochs": n_epochs, "w_physics": w_physics,
        "test_mse": test_metrics["mse"],
        "test_energy_err": test_metrics["energy_err"],
        "test_momentum_err": test_metrics["momentum_err"],
        "test_equivariance_err": test_metrics["equivariance_err"],
        "test_energy_drift": test_metrics["energy_drift"],
        "test_momentum_drift": test_metrics["momentum_drift"],
        "test_graph_topology_recall": test_metrics["graph_recovery"]["topology_recall"],
        "test_adjacency_auc": test_metrics["graph_recovery"]["adjacency_auc"],
        "test_graph_strength_corr": test_metrics["graph_recovery"]["strength_corr"],
        "gen_mse": gen_metrics["mse"],
        "gen_energy_err": gen_metrics["energy_err"],
        "gen_equivariance_err": gen_metrics["equivariance_err"],
        "mean_gate": test_metrics["mean_gate"],
        "wall_time_s": time.time() - t0,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  -> {out_dir}/{name}.json")
    _print_metrics(results)
    return results


def _print_metrics(r: dict):
    print(f"    test_mse={r['test_mse']:.5f}  energy_err={r['test_energy_err']:.6f} "
          f"momentum_err={r['test_momentum_err']:.6f}")
    print(f"    equivariance_err={r['test_equivariance_err']:.3e} "
          f"energy_drift={r['test_energy_drift']:.5f} "
          f"topo_recall={r['test_graph_topology_recall']} "
          f"adj_auc={r['test_adjacency_auc']} "
          f"strength_corr={r['test_graph_strength_corr']:.3f}")
    print(f"    gen_mse={r['gen_mse']:.5f}  mean_gate={r['mean_gate']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--only", type=str, default=None,
                    help="run a single experiment: gravity|spring|lennard_jones|ablation")
    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    out_dir = args.out
    n_epochs = args.epochs
    results = {}

    only = args.only

    if only in (None, "gravity"):
        # --- 1. N-body gravity (inverse-square attraction, Plummer-softened) ---
        gravity = NBodyGravity(n_particles=3, G=1.0, dt=0.01, n_steps=50)
        model = PhysRNet(n_particles=3, state_dim=4, physics_type="gravity", hidden=64)
        results["nbody_gravity"] = run_experiment(
            "nbody_gravity", gravity, model, out_dir, 300, 100, 100, n_epochs, 0.01, device)

    if only in (None, "spring"):
        # --- 2. Spring chain (Hooke's law, chain topology) ---
        spring = SpringChain(n_particles=4, k=10.0, L0=1.0, dt=0.005, n_steps=50)
        model = PhysRNet(n_particles=4, state_dim=4, physics_type="spring", hidden=64)
        results["spring_chain"] = run_experiment(
            "spring_chain", spring, model, out_dir, 300, 100, 100, n_epochs, 0.005, device)

    if only in (None, "reasoning_spring"):
        # --- 5. Reasoning-only ablation (gate frozen to 0): does the interpretable
        # module ALONE learn the force law and recover the interaction graph? ---
        spring = SpringChain(n_particles=4, k=10.0, L0=1.0, dt=0.005, n_steps=50)
        model = PhysRNet(n_particles=4, state_dim=4, physics_type="spring", hidden=64,
                         freeze_gate=0.0)
        results["reasoning_only_spring"] = run_experiment(
            "reasoning_only_spring", spring, model, out_dir, 300, 100, 100, n_epochs, 0.005, device)

    if only in (None, "reasoning_gravity"):
        gravity = NBodyGravity(n_particles=3, G=1.0, dt=0.01, n_steps=50)
        model = PhysRNet(n_particles=3, state_dim=4, physics_type="gravity", hidden=64,
                         freeze_gate=0.0)
        results["reasoning_only_gravity"] = run_experiment(
            "reasoning_only_gravity", gravity, model, out_dir, 300, 100, 100, n_epochs, 0.01, device)

    if only in (None, "lennard_jones"):
        # --- 3. Lennard-Jones fluid (6-12 potential) ---
        lj = LennardJones(n_particles=4, epsilon=1.0, sigma=1.0, dt=0.002, n_steps=50)
        model = PhysRNet(n_particles=4, state_dim=4, physics_type="lennard_jones", hidden=64)
        results["lennard_jones"] = run_experiment(
            "lennard_jones", lj, model, out_dir, 300, 100, 100, n_epochs, 0.002, device)

    if only in (None, "ablation"):
        # --- 4. PINN ablation: with vs without conservation losses ---
        gravity = NBodyGravity(n_particles=3, G=1.0, dt=0.01, n_steps=50)
        model_no_pinn = PhysRNet(n_particles=3, state_dim=4, physics_type="gravity", hidden=64)
        results["ablation_no_pinn"] = run_experiment(
            "ablation_no_pinn", gravity, model_no_pinn, out_dir,
            300, 100, 100, n_epochs, 0.01, device, w_physics=0.0)

    # --- summary ---
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  summary -> {out_dir}/summary.json")


if __name__ == "__main__":
    main()
