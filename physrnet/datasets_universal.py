"""Universal datasets for all physics domains.

Provides a single interface to generate data for all PSN-Universal domains.
Each domain returns a consistent dict format for training.

Domain list:
    1. gravity          — N-body gravity (softened)
    2. spring           — Spring chain
    3. lennard_jones    — Molecular dynamics
    4. fluid            — SPH fluid (Navier-Stokes)
    5. electromagnetism  — Charged particles
    6. quantum          — Schrödinger wavefunction
    7. heat             — Heat conduction
    8. relativistic     — Relativistic N-body
    9. thermo_ideal     — Thermodynamic processes
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Tuple

from .datasets_3d import (
    generate_nbody_3d,
    generate_spring_3d,
    generate_lennard_jones_3d,
)
from .physics.fluids import generate_sph_fluid
from .physics.electromagnetism import generate_charged_particles
from .physics.quantum import generate_quantum_system, generate_multiparticle_quantum
from .physics.thermodynamics import generate_heat_flow, generate_thermo_process
from .physics.relativistic import generate_relativistic_nbody


def make_universal_dataset(
    domain: str,
    n_samples: int = 100,
    n_particles: int = 100,
    n_steps: int = 50,
    seed: int = 42,
    **kwargs,
) -> Tuple[dict, dict, dict]:
    """Generate train/val/test splits for any physics domain.
    
    Returns:
        (train_data, val_data, test_data) — each is a dict with unified format:
            - 'pos': (S, T, N, 3) positions or grid
            - 'vel': (S, T, N, 3) velocities or field values
            - 'masses': (S, N) masses/weights
            - 'domain': str, physics domain name
            - 'params': dict, domain parameters
            - plus domain-specific keys (charges, T, V, psi, etc.)
    """
    n_train = int(n_samples * 0.7)
    n_val = int(n_samples * 0.15)
    n_test = n_samples - n_train - n_val
    
    datasets = []
    seeds = [seed, seed + n_train + 1, seed + n_train + n_val + 2]
    
    generators = {
        'gravity': lambda s: generate_nbody_3d(n_particles=n_particles, n_steps=n_steps, seed=s),
        'spring': lambda s: generate_spring_3d(n_particles=n_particles, n_steps=n_steps, seed=s),
        'lennard_jones': lambda s: generate_lennard_jones_3d(n_particles=n_particles, n_steps=n_steps, seed=s),
        'fluid': lambda s: generate_sph_fluid(n_particles=n_particles, n_steps=n_steps, seed=s),
        'electromagnetism': lambda s: generate_charged_particles(n_particles=n_particles, n_steps=n_steps, seed=s),
        'quantum': lambda s: generate_quantum_system(n_points=n_particles, n_steps=n_steps, seed=s, **kwargs),
        'heat': lambda s: generate_heat_flow(n_particles=n_particles, n_steps=n_steps, seed=s, **kwargs),
        'relativistic': lambda s: generate_relativistic_nbody(n_particles=n_particles, n_steps=n_steps, seed=s),
        'thermo_ideal': lambda s: generate_thermo_process(n_particles=n_particles, n_steps=n_steps, seed=s, **kwargs),
    }
    
    if domain not in generators:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(generators.keys())}")
    
    gen = generators[domain]
    
    for split_name, n_split in [('train', n_train), ('val', n_val), ('test', n_test)]:
        split_seed = seeds[['train', 'val', 'test'].index(split_name)]
        trajectories = []
        
        for i in range(max(n_split, 1)):
            traj = gen(split_seed + i)
            trajectories.append(traj)
        
        # Unify format
        unified = _unify_format(trajectories, domain)
        datasets.append(unified)
    
    return tuple(datasets)


def _unify_format(trajectories: List[dict], domain: str) -> dict:
    """Convert domain-specific trajectory dicts to unified format."""
    # All domains must provide 'pos'
    pos_key = 'pos' if 'pos' in trajectories[0] else 'positions'
    unified = {
        'pos': np.stack([t[pos_key] for t in trajectories]),
        'domain': domain,
        'params': trajectories[0].get('params', {}),
    }
    
    # Velocities / field values
    if 'vel' in trajectories[0]:
        unified['vel'] = np.stack([t['vel'] for t in trajectories])
    elif 'psi_real' in trajectories[0]:
        # Quantum: stack wavefunction as "velocity"
        unified['psi_real'] = np.stack([t['psi_real'] for t in trajectories])
        unified['psi_imag'] = np.stack([t['psi_imag'] for t in trajectories])
        unified['V'] = trajectories[0].get('V', np.zeros(trajectories[0]['psi_real'].shape[1]))
    elif 'T' in trajectories[0]:
        # Heat: stack temperature as "velocity"
        unified['T'] = np.stack([t['T'] for t in trajectories])
    
    # Masses / weights
    if 'masses' in trajectories[0]:
        masses = trajectories[0]['masses']
        if masses.ndim == 1:
            unified['masses'] = np.stack([
                np.broadcast_to(masses[None, :], (trajectories[0][pos_key].shape[0], len(masses)))
                for _ in trajectories
            ])
        else:
            unified['masses'] = np.stack([t['masses'] for t in trajectories])
    elif 'charges' in trajectories[0]:
        charges = trajectories[0]['charges']
        if charges.ndim == 1:
            S = np.prod(charges.shape) if hasattr(charges, 'shape') else len(charges)
            unified['masses'] = np.stack([
                np.broadcast_to(charges[None, :],
                                (trajectories[0][pos_key].shape[0], S))
                for _ in trajectories
            ])
        else:
            unified['masses'] = np.stack([t['charges'] for t in trajectories])
        unified['charges'] = unified['masses'].copy()
    else:
        # Default: unit masses
        S = trajectories[0][pos_key].shape[1] if trajectories[0][pos_key].ndim >= 2 else 1
        unified['masses'] = np.ones((len(trajectories),
                                      trajectories[0][pos_key].shape[0], S),
                                     dtype=np.float32)
    
    # Domain-specific extras
    if domain == 'electromagnetism':
        unified['charges'] = unified.get('charges', unified['masses'].copy())
    elif domain == 'thermo_ideal':
        if 'P' in trajectories[0]:
            unified['P'] = np.stack([t['P'] for t in trajectories])
        if 'V' in trajectories[0]:
            unified['volume'] = np.stack([t['V'] for t in trajectories])
    elif domain == 'heat':
        if 'boundary_mask' in trajectories[0]:
            unified['boundary_mask'] = trajectories[0]['boundary_mask']
    
    return unified


def get_domain_metadata() -> dict:
    """Return metadata for all supported domains.
    
    Used for documentation, ablation studies, and cross-domain analysis.
    """
    return {
        'gravity': {
            'category': 'particle',
            'equations': ['F = -GMm/r²', 'a = F/m'],
            'conserved': ['energy', 'momentum', 'angular_momentum'],
            'dimension': 3,
            'symmetry': 'E(3)',
            'key_paper': 'Sanchez-Gonzalez et al. (ICML 2020)',
            'real_world': 'orbital mechanics, astrophysics, satellite trajectories',
        },
        'spring': {
            'category': 'particle',
            'equations': ['F = -k(r - r0)', 'PE = ½k(r - r0)²'],
            'conserved': ['energy', 'momentum'],
            'dimension': 3,
            'symmetry': 'E(3)',
            'key_paper': 'Battaglia et al. (NeurIPS 2016)',
            'real_world': 'structural mechanics, molecular vibrations, vehicle suspensions',
        },
        'lennard_jones': {
            'category': 'particle',
            'equations': ['V = 4ε[(σ/r)¹² - (σ/r)⁶]', 'F = -∇V'],
            'conserved': ['energy', 'momentum'],
            'dimension': 3,
            'symmetry': 'E(3)',
            'key_paper': 'Behler & Parrinello (PRL 2007)',
            'real_world': 'molecular dynamics, protein folding, drug discovery',
        },
        'fluid': {
            'category': 'continuum',
            'equations': ['∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u', '∇·u = 0'],
            'conserved': ['mass', 'momentum', 'energy'],
            'dimension': 3,
            'symmetry': 'Galilean (approximately E(3))',
            'key_paper': 'Ummenhofer et al. (ICLR 2020)',
            'real_world': 'weather, aerodynamics, ocean currents, blood flow',
        },
        'electromagnetism': {
            'category': 'particle',
            'equations': ['∇·E = ρ/ε₀', '∇×B = μ₀J + μ₀ε₀∂E/∂t', 'F = q(E + v×B)'],
            'conserved': ['charge', 'energy', 'momentum'],
            'dimension': 3,
            'symmetry': 'Lorentz (Poincaré)',
            'key_paper': 'Raissi et al. (Science 2019)',
            'real_world': 'antenna design, particle accelerators, plasma physics',
        },
        'quantum': {
            'category': 'continuum',
            'equations': ['iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + Vψ'],
            'conserved': ['probability (|ψ|²)', 'energy (<H>)'],
            'dimension': 1,
            'symmetry': 'U(1) gauge',
            'key_paper': 'Carleo & Troyer (Science 2017)',
            'real_world': 'quantum computing, material band structure, chemical reactions',
        },
        'heat': {
            'category': 'continuum',
            'equations': ['∂T/∂t = α∇²T', 'q = -k∇T'],
            'conserved': ['energy (thermal)'],
            'dimension': 2,
            'symmetry': 'E(2) (rotations of the plate)',
            'key_paper': 'Cai et al. (2021)',
            'real_world': 'chip cooling, building insulation, engine thermal management',
        },
        'relativistic': {
            'category': 'particle',
            'equations': ['p = γmv', 'E = γmc²', 'dp/dt = F'],
            'conserved': ['energy-momentum', 'proper time'],
            'dimension': 3,
            'symmetry': 'Lorentz (Poincaré)',
            'key_paper': 'Greydanus et al. (NeurIPS 2019)',
            'real_world': 'GPS satellites, particle colliders, astrophysical jets',
        },
        'thermo_ideal': {
            'category': 'thermo',
            'equations': ['PV = nRT', 'dU = δQ - PdV', 'TV^(γ-1) = const (adiabatic)'],
            'conserved': ['internal energy', 'entropy (reversible)'],
            'dimension': 0,  # macroscopic state, not spatial
            'symmetry': 'none (state-space variables)',
            'key_paper': 'Raissi et al. (Science 2019)',
            'real_world': 'engines, refrigeration, power plants, HVAC systems',
        },
    }


def make_cross_domain_benchmarks(n_particles: int = 64, n_steps: int = 50,
                                  seed: int = 42) -> dict:
    """Generate standard benchmarks for all domains at once.
    
    Returns dict mapping domain -> (train, val, test) datasets.
    Used for cross-domain generalization studies.
    """
    benchmarks = {}
    
    for domain in PHYSICS_DOMAINS:
        try:
            benchmarks[domain] = make_universal_dataset(
                domain, n_particles=n_particles,
                n_steps=n_steps, seed=seed)
        except Exception as e:
            print(f"  Warning: {domain} generation failed: {e}")
            benchmarks[domain] = None
    
    return benchmarks


PHYSICS_DOMAINS = [
    'gravity', 'spring', 'lennard_jones',
    'fluid', 'electromagnetism', 'quantum',
    'heat', 'relativistic', 'thermo_ideal',
]