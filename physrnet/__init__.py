"""PSN-1: Physics Systems Network.

Combines E(3)-equivariant encoders, attention-based reasoning,
conservation law discovery, and physics-informed constraints into a
single architecture for learning and understanding physical systems.

Modules:
    equivariant.py       — E(2)-equivariant message passing (v1)
    e3_equivariant.py    — E(3)-equivariant message passing (v2)
    pinn.py              — physics constraint losses
    reasoning.py         — graph-based interaction reasoning (v1)
    attention_reasoning.py — multi-head attention reasoning (v2)
    conservation.py      — conservation law discovery
    model.py             — PhysRNet v1 assembly
    model_v2.py          — PSN-1 v2 assembly (NMI quality)
    datasets.py          — 2D physics trajectory generation
    datasets_3d.py       — 3D physics trajectory generation
    training.py          — training loops (v1)
    training_v2.py       — training loops with ablations (v2)
"""

from .model import PhysRNet  # noqa: F401
from .model_v2 import PSN1v2  # noqa: F401
from .pinn import PhysicsResidual  # noqa: F401
from .conservation import PhysicsDiscovery  # noqa: F401

__version__ = "0.2.0"
