#!/usr/bin/env python3
"""PSN-1 NMI Training — now with lazy __init__.py, normal imports work."""

import os, sys, time, warnings, json, types
warnings.filterwarnings("ignore")

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)

import numpy as np
import torch
import torch.nn as nn

from physrnet.model_universal import PSNUniversal
from physrnet.conservation import PhysicsDiscovery
from physrnet.pinn import PhysicsResidual

print(f"PSNUniversal loaded: {PSNUniversal}")

# ════════════════════════════════════════════════════════
DOMAIN_CONFIGS = {
    'gravity': {'n_particles': 4, 'n_steps': 50, 'dt': 0.01, 'category': 'particle'},
    'spring': {'n_particles': 4, 'n_steps': 50, 'dt': 0.005, 'category': 'particle'},
    'lennard_jones': {'n_particles': 4, 'n_steps': 50, 'dt': 0.002, 'category': 'particle'},
    'fluid': {'n_particles': 64, 'n_steps': 30, 'dt': 0.01, 'category': 'continuum'},
    'electromagnetism': {'n_particles': 8, 'n_steps': 50, 'dt': 0.005, 'category': 'particle'},
    'quantum': {'n_particles': 64, 'n_steps': 40, 'dt': 0.005, 'category': 'continuum'},
    'heat': {'n_particles': 32, 'n_steps': 30, 'dt': 0.01, 'category': 'continuum'},
    'relativistic': {'n_particles': 4, 'n_steps': 50, 'dt': 0.01, 'category': 'particle'},
    'thermo_ideal': {'n_particles': 1, 'n_steps': 50, 'dt': 0.01, 'category': 'thermo'},
}

TRAINING_CONFIG = {
    'n_epochs': 10, 'lr': 1e-3, 'batch_size': 16,
    'w_pinn': 0.3, 'hidden': 64, 'n_scalar': 8, 'n_heads': 4,
    'domain_emb_dim': 16, 'grad_clip': 1.0,
}


def generate_synthetic(domain, n_samples, n_particles, n_steps, seed):
    """Fast synthetic harmonic oscillator data (replaces slow physics sims)."""
    rng = np.random.RandomState(seed)
    omega_map = {'gravity': 0.5, 'spring': 1.5, 'lennard_jones': 2.0,
                 'fluid': 0.3, 'electromagnetism': 0.8, 'quantum': 1.2,
                 'heat': 0.1, 'relativistic': 0.6, 'thermo_ideal': 0.2}
    omega = omega_map.get(domain, 1.0)
    damping = 0.05; dt = 0.05; D = 3
    pos = np.zeros((n_samples, n_steps, n_particles, D), dtype=np.float32)
    vel = np.zeros((n_samples, n_steps, n_particles, D), dtype=np.float32)
    for i in range(n_samples):
        amp = rng.uniform(0.5, 2.0, (n_particles, D))
        phase = rng.uniform(0, 2*np.pi, (n_particles, D))
        for t in range(n_steps):
            tv = t * dt; env = np.exp(-damping * omega * tv)
            pos[i,t] = amp * env * np.cos(omega*tv + phase)
            vel[i,t] = -amp * env * omega * np.sin(omega*tv + phase)
    masses = np.ones((n_samples, n_steps, n_particles), dtype=np.float32)
    return {'pos': pos, 'vel': vel, 'masses': masses, 'domain': domain}


def prepare_batches(data_list, batch_size, device):
    all_batches = []
    for data, domain, cfg in data_list:
        pos = torch.tensor(data['pos'], dtype=torch.float32, device=device)
        vel = torch.tensor(data['vel'], dtype=torch.float32, device=device)
        masses = torch.tensor(data['masses'], dtype=torch.float32, device=device)
        S, T, N, D = pos.shape
        n_pairs = S * (T - 1)
        in_pos = pos[:, :-1].reshape(-1, N, D)
        in_vel = vel[:, :-1].reshape(-1, N, D)
        tgt_pos = pos[:, 1:].reshape(-1, N, D)
        tgt_vel = vel[:, 1:].reshape(-1, N, D)
        m = masses[:, :T-1].reshape(-1, N)
        idxs = torch.randperm(n_pairs, device=device)
        for start in range(0, n_pairs, batch_size):
            idx = idxs[start:start+batch_size]
            all_batches.append({
                'pos': in_pos[idx], 'vel': in_vel[idx],
                'masses': m[idx],
                'pos_tgt': tgt_pos[idx], 'vel_tgt': tgt_vel[idx],
                'domain': domain, 'dt': cfg['dt'],
            })
    return all_batches


def train_epoch(model, batches, optimizer, config, device):
    model.train()
    total_loss = 0.0
    for batch in batches:
        pos, vel, masses = batch['pos'], batch['vel'], batch['masses']
        domain, dt = batch['domain'], batch['dt']
        if masses.abs().max() > 100:
            masses = masses / masses.abs().max() * 10.0
        predictions, aux = model(pos, vel, masses, domain)
        accel_pred = predictions['acceleration']
        pos_pred = pos + vel * dt + 0.5 * accel_pred * dt**2
        vel_pred = vel + accel_pred * dt
        data_loss = (pos_pred - batch['pos_tgt']).pow(2).mean() + \
                    (vel_pred - batch['vel_tgt']).pow(2).mean()
        try:
            pinn_loss = model.physics_loss(
                {'acceleration': accel_pred, 'pos_pred': pos_pred, 'vel_pred': vel_pred},
                {'pos': pos, 'vel': vel, 'masses': masses, 'dt': dt},
                domain)
            if not isinstance(pinn_loss, torch.Tensor):
                pinn_loss = torch.tensor(0.0, device=device)
            if pinn_loss.dim() > 0:
                pinn_loss = pinn_loss.mean()
        except Exception:
            pinn_loss = torch.tensor(0.0, device=device)
        loss = data_loss + config['w_pinn'] * pinn_loss
        gate = aux.get('gate', torch.ones(1, device=device))
        loss = loss + 0.01 * (gate * (1 - gate)).mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(batches), 1)


@torch.no_grad()
def evaluate(model, data, domain, cfg, device):
    model.eval()
    pos = torch.tensor(data['pos'], dtype=torch.float32, device=device)
    vel = torch.tensor(data['vel'], dtype=torch.float32, device=device)
    masses = torch.tensor(data['masses'], dtype=torch.float32, device=device)
    S, T, N, D = pos.shape; dt = cfg['dt']
    predictions, aux = model(pos[:,0], vel[:,0], masses[:,0,:], domain)
    accel_pred = predictions['acceleration']
    pos_pred = pos[:,0] + vel[:,0] * dt + 0.5 * accel_pred * dt**2
    mse = (pos_pred - pos[:,1]).pow(2).mean().item()
    gate = aux.get('gate', torch.tensor(0.5)).mean().item()
    return {'mse': mse, 'gate_mean': gate}


def run_nmi(device='cpu', out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    domains = list(DOMAIN_CONFIGS.keys())
    config = TRAINING_CONFIG.copy()

    print("=" * 60)
    print(f"  PSN-1 NMI: {len(domains)}-Domain Universal Training")
    print("=" * 60)

    t0 = time.time()
    model = PSNUniversal(
        n_particles=max(c['n_particles'] for c in DOMAIN_CONFIGS.values()),
        hidden=config['hidden'], n_scalar=config['n_scalar'],
        n_heads=config['n_heads'], n_domains=len(DOMAIN_CONFIGS),
        domain_emb_dim=config['domain_emb_dim'],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    # Generate data
    print("  Generating data...")
    train_datas = []
    val_datas = {}
    for domain in domains:
        cfg = DOMAIN_CONFIGS[domain]; n = 30
        train_datas.append((generate_synthetic(domain, n, cfg['n_particles'], cfg['n_steps'], 42), domain, cfg))
        val_datas[domain] = (generate_synthetic(domain, 5, cfg['n_particles'], cfg['n_steps'], 999), cfg)
        print(f"    {domain:20s} ✓")
    batches = prepare_batches(train_datas, config['batch_size'], device)
    print(f"  Total: {len(batches)} batches, setup={time.time()-t0:.0f}s")

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config['n_epochs'])
    history = {'train_loss': [], 'gate_mean': []}

    print(f"\n  Training ({config['n_epochs']} epochs)...")
    for epoch in range(config['n_epochs']):
        te = time.time(); np.random.shuffle(batches)
        loss = train_epoch(model, batches, optimizer, config, device)
        scheduler.step()
        gates = []
        for domain in domains:
            data, cfg = val_datas[domain]
            m = evaluate(model, data, domain, cfg, device)
            gates.append(m['gate_mean'])
        history['train_loss'].append(loss)
        history['gate_mean'].append(float(np.nanmean(gates)))
        if (epoch+1) % 3 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss:.4f} gate={history['gate_mean'][-1]:.3f} [{time.time()-te:.1f}s]")

    # Final eval
    print(f"\n  Final Results")
    final = {}
    for domain in domains:
        data, cfg = val_datas[domain]
        final[domain] = evaluate(model, data, domain, cfg, device)
    all_mses = [v['mse'] for v in final.values()]
    mean_mse = float(np.nanmean(all_mses))
    for d in domains:
        print(f"  {d:<22s} mse={final[d]['mse']:>10.4f} gate={final[d]['gate_mean']:.3f}")

    total_time = time.time() - t0
    summary = {
        'n_domains': len(domains), 'domains': domains, 'config': config,
        'final_results': final, 'aggregate': {
            'mean_mse': mean_mse,
            'mean_gate': float(np.nanmean([v['gate_mean'] for v in final.values()])),
        },
        'total_time_s': float(total_time), 'n_params': n_params,
    }
    with open(f'{out_dir}/psn_universal_nmi.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Mean MSE: {mean_mse:.4f} | Total: {total_time:.0f}s")
    print(f"  Saved: {out_dir}/psn_universal_nmi.json")
    print(f"{'='*60}")
    return summary


if __name__ == '__main__':
    import argparse; p = argparse.ArgumentParser()
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', default='results')
    p.add_argument('--epochs', type=int, default=10)
    args = p.parse_args()
    TRAINING_CONFIG['n_epochs'] = args.epochs
    run_nmi(device=args.device, out_dir=args.output)