"""Physics domain modules for PSN-Universal.

Each module implements one physical domain with:
  * Physics residuals (PINN losses)
  * Data generators for training/evaluation
  * Domain-specific conservation laws

Domains:
    fluids.py           — Navier-Stokes, SPH fluid dynamics
    electromagnetism.py — Maxwell's equations, charged particles
    quantum.py          — Schrödinger equation, quantum dynamics
    thermodynamics.py   — Heat equation, thermodynamic processes
    relativistic.py     — Special/General relativistic mechanics

References:
  * Sanchez-Gonzalez et al. (ICML 2020) — "Learning to Simulate Complex Physics"
  * Pfaff et al. (ICLR 2021) — "Learning Mesh-Based Simulation with Graph Networks"
  * Raissi et al. (Science 2019) — "Physics-informed neural networks"
  * Brandstetter et al. (NeurIPS 2022) — "Geometric and Physical Quantities improve E(3) GNNs"
  * Satorras et al. (ICML 2021) — "E(n) Equivariant Graph Neural Networks"
"""

from .fluids import NavierStokesResidual, generate_sph_fluid
from .electromagnetism import MaxwellResidual, generate_charged_particles
from .quantum import SchrodingerResidual, generate_quantum_system, generate_multiparticle_quantum
from .thermodynamics import HeatEquationResidual, ThermodynamicsResidual, generate_heat_flow, generate_thermo_process
from .relativistic import RelativisticResidual, GRGeodesicResidual, generate_relativistic_nbody

__all__ = [
    'NavierStokesResidual', 'generate_sph_fluid',
    'MaxwellResidual', 'generate_charged_particles',
    'SchrodingerResidual', 'generate_quantum_system', 'generate_multiparticle_quantum',
    'HeatEquationResidual', 'ThermodynamicsResidual', 'generate_heat_flow', 'generate_thermo_process',
    'RelativisticResidual', 'GRGeodesicResidual', 'generate_relativistic_nbody',
]