"""Full NMI experiment suite for PSN-1.

Runs:
  1. Three physics systems (gravity, spring, LJ) in 3D
  2. Equivariance verification
  3. Ablation study (5 variants)
  4. Conservation law discovery
  5. Generalization to unseen masses
  6. Rollout drift evaluation
"""

from __future__ import annotations

import time
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def run_system_experiment(physics_type, n_particles=5, n_epochs=40,
                          n_train=100, n_val=30, n_test=30,
                          device='cpu', out_dir='results'):
    """Run experiment on one physics system."""
    from physrnet.datasets_3d import make_dataset
    from physrnet.model_v2 import PSN1v2
    from physrnet.training_v2 import train_psn1v2

    print(f"\n{'='*60}")
    print(f"  System: {physics_type} (3D, {n_particles} particles)")
    print(f"{'='*60}")

    t0 = time.time()
    train_data, val_data, test_data = make_dataset(
        physics_type, n_particles=n_particles,
        n_train=n_train, n_val=n_val, n_test=n_test,
        n_steps=50, seed=42)
    print(f"  Data generated in {time.time()-t0:.1f}s")

    model = PSN1v2(n_particles=n_particles, physics_type=physics_type,
                   hidden=128, n_scalar=8, n_heads=4, use_conservation=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    results = train_psn1v2(model, train_data, val_data,
                           n_epochs=n_epochs, w_pinn=0.5,
                           device=device, out_dir=out_dir,
                           name=f'nmi_{physics_type}')

    # Test on held-out data
    model.eval()
    test_pos, test_vel, test_mass_raw = _to_torch(test_data, device)
    N = test_pos.shape[2]
    # Mass: (S, N, N) -> (S, N) per-particle
    test_mass = test_mass_raw[:, 0, :] if test_mass_raw.dim() == 3 else test_mass_raw

    with torch.no_grad():
        # Single-step prediction on test set
        in_pos = test_pos[:, :-1].reshape(-1, N, 3)
        in_vel = test_vel[:, :-1].reshape(-1, N, 3)
        tgt_pos = test_pos[:, 1:].reshape(-1, N, 3)
        T_test = test_pos.shape[1]
        m = test_mass.unsqueeze(1).expand(-1, T_test - 1, -1).reshape(-1, N)

        accel, attn, edge_w, gate = model(in_pos, in_vel, m)
        dt = 0.01
        pos_pred = in_pos + in_vel * dt + 0.5 * accel * dt**2
        test_mse = (pos_pred - tgt_pos).pow(2).mean().item()

    results['test_mse'] = test_mse
    print(f"  Test MSE: {test_mse:.6f}")

    # Generalization to unseen masses
    gen_results = test_generalization(model, physics_type, n_particles, device)
    results['generalization'] = gen_results

    return results


def test_generalization(model, physics_type, n_particles, device):
    """Test generalization to different masses."""
    from physrnet.datasets_3d import make_dataset

    # Generate test with different random seed (= different masses)
    result = make_dataset(physics_type, n_particles=n_particles,
                          n_train=20, n_val=0, n_test=0, n_steps=50, seed=999)
    gen_data = result[0]  # first element is the training split with new seed

    gen_pos, gen_vel, gen_mass_raw = _to_torch(gen_data, device)
    N = gen_pos.shape[2]
    gen_mass = gen_mass_raw[:, 0, :] if gen_mass_raw.dim() == 3 else gen_mass_raw

    model.eval()
    with torch.no_grad():
        in_pos = gen_pos[:, :-1].reshape(-1, N, 3)
        in_vel = gen_vel[:, :-1].reshape(-1, N, 3)
        tgt_pos = gen_pos[:, 1:].reshape(-1, N, 3)
        T_gen = gen_pos.shape[1]
        m = gen_mass.unsqueeze(1).expand(-1, T_gen - 1, -1).reshape(-1, N)

        accel, _, _, _ = model(in_pos, in_vel, m)
        dt = 0.01
        pos_pred = in_pos + in_vel * dt + 0.5 * accel * dt**2
        gen_mse = (pos_pred - tgt_pos).pow(2).mean().item()

    print(f"  Generalization MSE: {gen_mse:.6f}")
    return {'gen_mse': gen_mse}


def run_ablation(physics_type='gravity', n_particles=5, n_epochs=30,
                 device='cpu', out_dir='results'):
    """Run ablation study on one system."""
    from physrnet.datasets_3d import make_dataset
    from physrnet.model_v2 import PSN1v2
    from physrnet.training_v2 import train_psn1v2

    print(f"\n{'='*60}")
    print(f"  Ablation study: {physics_type}")
    print(f"{'='*60}")

    train_data, val_data, _ = make_dataset(
        physics_type, n_particles=n_particles,
        n_train=80, n_val=20, n_test=0, n_steps=50, seed=42)

    configs = [
        ('full', None, 0.5),
        ('no_pinn', None, 0.0),
        ('equiv_only', 1.0, 0.5),
        ('reason_only', 0.0, 0.5),
    ]

    results = {}
    for name, freeze, w_pinn in configs:
        print(f"\n  --- {name} ---")
        model = PSN1v2(n_particles=n_particles, physics_type=physics_type,
                       use_conservation=(name != 'no_conservation'))
        r = train_psn1v2(model, train_data, val_data,
                         n_epochs=n_epochs, w_pinn=w_pinn,
                         device=device, out_dir=out_dir,
                         name=f'ablation_{physics_type}_{name}',
                         freeze_gate=freeze)
        results[name] = {
            'mse': r['test_mse'],
            'equiv_err': r['test_equivariance_err'],
            'drift': r['test_drift'],
            'gate': r['mean_gate'],
        }

    return results


def _to_torch(data_dict, device):
    pos = torch.tensor(data_dict['pos'], dtype=torch.float32, device=device)
    vel = torch.tensor(data_dict['vel'], dtype=torch.float32, device=device)
    masses = torch.tensor(data_dict['masses'], dtype=torch.float32, device=device)
    return pos, vel, masses


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--n-particles', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--out-dir', type=str, default='results')
    parser.add_argument('--skip-ablation', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}

    # Run all three systems
    for physics in ['gravity', 'spring', 'lennard_jones']:
        r = run_system_experiment(physics, n_particles=args.n_particles,
                                   n_epochs=args.epochs, device=args.device,
                                   out_dir=args.out_dir)
        all_results[physics] = r

    # Ablation study
    if not args.skip_ablation:
        for physics in ['gravity', 'spring']:
            ablation = run_ablation(physics, n_particles=args.n_particles,
                                     n_epochs=args.epochs // 2,
                                     device=args.device, out_dir=args.out_dir)
            all_results[f'ablation_{physics}'] = ablation

    # Save summary
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'history'}
               for k, v in all_results.items()}
    with open(f'{args.out_dir}/nmi_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        if isinstance(v, dict) and 'test_mse' in v:
            print(f"  {k}: MSE={v['test_mse']:.6f} equiv={v['test_equivariance_err']:.2e} "
                  f"drift={v['test_drift']:.4f}")


if __name__ == '__main__':
    main()
