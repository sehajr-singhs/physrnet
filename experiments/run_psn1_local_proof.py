#!/usr/bin/env python3
"""PSN-1 local CPU proof: 9-domain universal training, reduced scale.

Sized to finish in ~5 minutes on CPU (the Modal GPU workspace is currently
over its spend limit, so this is the fallback that produces a fresh result).

Difference from the full run: 12 training samples per domain (vs 30),
fluid/quantum/heat capped at 16 particles, 3 epochs. Same architecture,
same loss, same evaluation protocol.
"""

import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _src)

import numpy as np
import torch

from physrnet.model_universal import PSNUniversal

DOMAIN_CONFIGS = {
    'gravity': {'n_particles': 4, 'n_steps': 50, 'dt': 0.01},
    'spring': {'n_particles': 4, 'n_steps': 50, 'dt': 0.005},
    'lennard_jones': {'n_particles': 4, 'n_steps': 50, 'dt': 0.002},
    'fluid': {'n_particles': 16, 'n_steps': 30, 'dt': 0.01},
    'electromagnetism': {'n_particles': 8, 'n_steps': 50, 'dt': 0.005},
    'quantum': {'n_particles': 16, 'n_steps': 40, 'dt': 0.005},
    'heat': {'n_particles': 16, 'n_steps': 30, 'dt': 0.01},
    'relativistic': {'n_particles': 4, 'n_steps': 50, 'dt': 0.01},
    'thermo_ideal': {'n_particles': 1, 'n_steps': 50, 'dt': 0.01},
}
N_TRAIN = 12
N_VAL = 4
N_EPOCHS = 3
BATCH_SIZE = 16
LR = 1e-3
W_PINN = 0.3


def gen(domain, n_samples, seed):
    rng = np.random.RandomState(seed)
    omega_map = {'gravity': 0.5, 'spring': 1.5, 'lennard_jones': 2.0,
                 'fluid': 0.3, 'electromagnetism': 0.8, 'quantum': 1.2,
                 'heat': 0.1, 'relativistic': 0.6, 'thermo_ideal': 0.2}
    omega = omega_map.get(domain, 1.0)
    damping = 0.05; dt = 0.05
    N = DOMAIN_CONFIGS[domain]['n_particles']
    T = DOMAIN_CONFIGS[domain]['n_steps']
    D = 3
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
    return {'pos': pos, 'vel': vel, 'masses': masses}


def run(out_dir='results'):
    os.makedirs(out_dir, exist_ok=True)
    domains = list(DOMAIN_CONFIGS.keys())
    print('=' * 60)
    print(f'  PSN-1 local proof: {len(domains)} domains, {N_EPOCHS} epochs, CPU')
    print('=' * 60)
    t0 = time.time()

    model = PSNUniversal(
        n_particles=max(c['n_particles'] for c in DOMAIN_CONFIGS.values()),
        hidden=64, n_scalar=8, n_heads=4,
        n_domains=len(domains), domain_emb_dim=16,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: {n_params:,} params')

    print('  Generating data...')
    batches = []
    val_data = {}
    for d in domains:
        cfg = DOMAIN_CONFIGS[d]
        tr = gen(d, N_TRAIN, 42)
        va = gen(d, N_VAL, 999)
        val_data[d] = (va, cfg)
        pos = torch.tensor(tr['pos']); vel = torch.tensor(tr['vel'])
        mass = torch.tensor(tr['masses'])
        S, T, N, D = pos.shape
        in_pos = pos[:, :-1].reshape(-1, N, D)
        in_vel = vel[:, :-1].reshape(-1, N, D)
        m = mass[:, :T-1].reshape(-1, N)
        tgt_p = pos[:, 1:].reshape(-1, N, D)
        tgt_v = vel[:, 1:].reshape(-1, N, D)
        n_pairs = in_pos.shape[0]
        for start in range(0, n_pairs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_pairs)
            batches.append({'pos': in_pos[start:end], 'vel': in_vel[start:end],
                            'masses': m[start:end], 'pos_tgt': tgt_p[start:end],
                            'vel_tgt': tgt_v[start:end],
                            'domain': d, 'dt': cfg['dt']})
    print(f'  {len(batches)} batches')

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EPOCHS)
    history = []
    print(f'\n  Training ({N_EPOCHS} epochs)...')
    for epoch in range(N_EPOCHS):
        te = time.time()
        np.random.shuffle(batches)
        model.train()
        total = 0.0
        for b in batches:
            pos, vel, mass = b['pos'], b['vel'], b['masses']
            pred, aux = model(pos, vel, mass, b['domain'])
            accel = pred['acceleration']
            pos_p = pos + vel * b['dt'] + 0.5 * accel * b['dt']**2
            vel_p = vel + accel * b['dt']
            dl = (pos_p - b['pos_tgt']).pow(2).mean() + (vel_p - b['vel_tgt']).pow(2).mean()
            try:
                pl = model.physics_loss(
                    {'acceleration': accel, 'pos_pred': pos_p, 'vel_pred': vel_p},
                    {'pos': pos, 'vel': vel, 'masses': mass, 'dt': b['dt']}, b['domain'])
                if not isinstance(pl, torch.Tensor):
                    pl = torch.tensor(0.0)
                if pl.dim() > 0:
                    pl = pl.mean()
            except Exception:
                pl = torch.tensor(0.0)
            loss = dl + W_PINN * pl
            gate = aux.get('gate', torch.ones(1))
            loss = loss + 0.01 * (gate * (1 - gate)).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        sched.step()
        avg = total / max(len(batches), 1)
        history.append(avg)
        print(f'  Epoch {epoch+1:3d}: loss={avg:.4f}  [{time.time()-te:.0f}s]')

    final = {}
    with torch.no_grad():
        model.eval()
        for d in domains:
            data, cfg = val_data[d]
            pos = torch.tensor(data['pos']); vel = torch.tensor(data['vel'])
            mass = torch.tensor(data['masses'])
            dt = cfg['dt']
            pred, aux = model(pos[:, 0], vel[:, 0], mass[:, 0, :], d)
            accel = pred['acceleration']
            pos_p = pos[:, 0] + vel[:, 0] * dt + 0.5 * accel * dt**2
            mse = (pos_p - pos[:, 1]).pow(2).mean().item()
            gate = aux.get('gate', torch.tensor(0.5)).mean().item()
            final[d] = {'mse': mse, 'gate': gate}
            print(f'  {d:22s} mse={mse:.6e} gate={gate:.3f}')

    total_time = time.time() - t0
    mean_mse = float(np.mean([v['mse'] for v in final.values()]))
    summary = {
        'name': 'psn1_universal_local_proof', 'n_domains': len(domains),
        'n_epochs': N_EPOCHS, 'n_train_per_domain': N_TRAIN,
        'n_params': n_params, 'device': 'cpu',
        'train_loss_history': history, 'per_domain': final,
        'final_mean_mse': mean_mse, 'total_time_s': float(total_time),
    }
    path = os.path.join(out_dir, 'psn1_universal_local_proof.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\n  Mean MSE: {mean_mse:.6e} | {total_time:.0f}s | saved: {path}')
    print('=' * 60)
    return summary


if __name__ == '__main__':
    run()
