"""PSN-1: Physics Systems Network → PSN-Universal.

Combines E(3)-equivariant encoders, attention-based reasoning,
conservation law discovery, and physics-informed constraints into a
single architecture for learning and understanding ALL physical systems.

Core modules:
    equivariant.py         — E(2)-equivariant message passing (v1)
    e3_equivariant.py      — E(3)-equivariant message passing (v2)
    pinn.py                — physics constraint losses
    reasoning.py           — graph-based interaction reasoning (v1)
    attention_reasoning.py — multi-head attention reasoning (v2)
    conservation.py        — conservation law discovery
    model.py               — PhysRNet v1 assembly
    model_v2.py            — PSN-1 v2 assembly
    model_universal.py     — PSN-Universal (all 9 physics domains)
    datasets.py            — 2D physics trajectory generation
    datasets_3d.py         — 3D physics trajectory generation (original 3 domains)
    datasets_universal.py  — Universal datasets for all 9 domains
    training.py            — training loops (v1)
    training_v2.py         — training loops with ablations (v2)

Physics domains (physics/):
    fluids.py              — Navier-Stokes, SPH fluid dynamics
    electromagnetism.py    — Maxwell's equations, charged particles
    quantum.py             — Schrödinger equation, quantum dynamics
    thermodynamics.py      — Heat equation, thermodynamic processes
    relativistic.py        — Special/General relativistic mechanics

Covers 9 domains: gravity, spring, lennard_jones, fluid, electromagnetism,
quantum, heat, relativistic, thermo_ideal.
"""

# All imports are lazy to avoid heavy module cascading at package init.
# Use explicit imports like: from physrnet.model_universal import PSNUniversal

__version__ = "0.3.0"

def __getattr__(name):
    """Lazy import for package-level names."""
    if name == 'PhysRNet':
        from .model import PhysRNet
        return PhysRNet
    if name == 'PSN1v2':
        from .model_v2 import PSN1v2
        return PSN1v2
    if name == 'PSNUniversal':
        from .model_universal import PSNUniversal
        return PSNUniversal
    if name == 'PhysicsResidual':
        from .pinn import PhysicsResidual
        return PhysicsResidual
    if name == 'PhysicsDiscovery':
        from .conservation import PhysicsDiscovery
        return PhysicsDiscovery
    raise AttributeError(f"module 'physrnet' has no attribute '{name}'")
