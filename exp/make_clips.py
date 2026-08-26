"""Render PSN-1 simulation clips with pure numpy + PIL (no torch, no matplotlib).

Ground-truth trajectories from the physics simulators (N-body gravity, spring
chain, Lennard-Jones) and the nine synthetic harmonic-domain generators are
rendered as animated GIFs into ../results/.

Clip files produced (matching the site's grid):
  clip_gravity.gif clip_spring.gif clip_lennard_jones.gif clip_fluid.gif
  clip_electromagnetism.gif clip_quantum.gif clip_heat.gif clip_relativistic.gif
  clip_thermo_ideal.gif clip_real_gravity.gif clip_real_spring.gif clip_real_lennard_jones.gif
"""
from __future__ import annotations
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from physrnet.datasets import NBodyGravity, SpringChain, LennardJones

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
os.makedirs(OUT, exist_ok=True)

W, H = 380, 360
PAD = 28


def _norm(xy):
    xmin, ymin = xy.reshape(-1, 2).min(0)
    xmax, ymax = xy.reshape(-1, 2).max(0)
    span = max(xmax - xmin, ymax - ymin, 1e-6)
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    s = (min(W, H) - 2 * PAD) / span
    return (xy - np.array([cx, cy])) * s + np.array([W / 2, H / 2])


def _render(traj, path, title, colors, trail=14, radius=6):
    """traj: (T, N, 2) already normalized. One colour per particle."""
    T, N, _ = traj.shape
    n_frames = min(T, 40)
    frames = []
    for j in range(n_frames):
        k = int(j / (n_frames - 1) * (T - 1)) if n_frames > 1 else 0
        img = Image.new("RGB", (W, H + 26), (11, 15, 20))
        d = ImageDraw.Draw(img)
        d.text((10, H + 6), title, fill=(148, 163, 184))
        t0 = max(0, k - trail)
        for n in range(N):
            col = colors[n % len(colors)]
            pts = [tuple(traj[t, n]) for t in range(t0, k + 1)]
            if len(pts) > 1:
                d.line(pts, fill=col + (90,), width=2)
            x, y = traj[k, n]
            d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=col)
        frames.append(img)
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=90, loop=0)
    print(f"  {os.path.basename(path)}  {T} steps", flush=True)


COLORS = [(56, 189, 248), (167, 139, 250), (74, 222, 128),
          (251, 146, 60), (244, 114, 182), (250, 204, 21),
          (148, 163, 184), (96, 165, 250)]


def real_clips():
    sims = [
        ("real_gravity", NBodyGravity(n_particles=3, dt=0.01, n_steps=70), "gravity · 3 bodies · RK4"),
        ("real_spring", SpringChain(n_particles=4, dt=0.005, n_steps=70), "springs · 4 masses · RK4"),
        ("real_lennard_jones", LennardJones(n_particles=4, dt=0.002, n_steps=70), "Lennard-Jones · 4 particles"),
    ]
    for name, sim, label in sims:
        data = sim.generate(n_traj=1, seed=123)
        xy = _norm(data["positions"][0][:, :, :2])
        _render(xy, os.path.join(OUT, f"clip_{name}.gif"), label, COLORS)


OMEGA = {"gravity": 0.5, "spring": 1.5, "lennard_jones": 2.0, "fluid": 0.3,
         "electromagnetism": 0.8, "quantum": 1.2, "heat": 0.1,
         "relativistic": 0.6, "thermo_ideal": 0.2}
N_PART = {"gravity": 4, "spring": 4, "lennard_jones": 4, "fluid": 64,
          "electromagnetism": 8, "quantum": 64, "heat": 32,
          "relativistic": 4, "thermo_ideal": 1}


def synthetic_clips():
    rng = np.random.RandomState(777)
    for domain, omega in OMEGA.items():
        N = N_PART[domain]
        T = 50
        dt = 0.05
        amp = rng.uniform(0.5, 2.0, (N, 2))
        phase = rng.uniform(0, 2 * np.pi, (N, 2))
        traj = np.zeros((T, N, 2))
        for t in range(T):
            tv = t * dt
            env = np.exp(-0.05 * omega * tv)
            traj[t] = amp * env * np.cos(omega * tv + phase)
        xy = _norm(traj)
        _render(xy, os.path.join(OUT, f"clip_{domain}.gif"),
                f"{domain} · damped harmonic · N={N}", COLORS)


if __name__ == "__main__":
    real_clips()
    synthetic_clips()
    print("done", flush=True)
