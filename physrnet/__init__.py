"""PhysRNet: Physics-Aware Reasoning Network.

Combines E(3)-equivariant encoders, physics-informed constraints, and
graph-based reasoning modules into a single architecture for learning
and understanding physical systems.

Modules:
    equivariant.py  — E(3)-equivariant message passing
    pinn.py         — physics constraint losses (energy, momentum conservation)
    reasoning.py    — graph-based interaction reasoning (GNOmE-style)
    model.py        — full PhysRNet assembly
    datasets.py     — synthetic physics trajectory generation
    training.py     — training and evaluation loops
"""

from .model import PhysRNet  # noqa: F401
from .pinn import PhysicsResidual  # noqa: F401
from .datasets import NBodyGravity, SpringChain, LennardJones  # noqa: F401

__version__ = "0.1.0"
