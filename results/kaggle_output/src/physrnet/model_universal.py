"""PSN-Universal: One architecture for all physics domains.

The key insight: every physical system is a graph of interacting entities.
Whether it's particles under gravity, fluid elements under pressure,
charged particles under EM forces, or wavefunction grids under the
Schrodinger equation — they're all graphs with:
  * Nodes: entities with state (position, velocity, charge, temperature, ...)
  * Edges: interactions governed by physical laws
  * Conservation: energy, momentum, charge, probability, ...

PSN-Universal uses a *single* unified architecture with:
  1. E(3)-equivariant encoder (works for all spatial systems)
  2. Domain-conditioned attention reasoning
  3. Universal conservation law discovery
  4. Multi-domain PINN loss
  5. Learned domain embedding for cross-domain transfer

This directly competes with Project Prometheus ($38B, Bezos-backed)
which claims to "understand the laws of physics." PSN-Universal covers
8 domains in one framework:

  Classical:  gravity, springs, Lennard-Jones
  Continuum:  fluid dynamics (Navier-Stokes), heat equation
  Field:      electromagnetism (Maxwell)
  Quantum:    Schrödinger equation
  Relativistic: special relativity, GR geodesics
  Thermo:     thermodynamic processes (ideal gas)

References:
  * Sanchez-Gonzalez et al. (ICML 2020) — GNNs for physics simulation
  * Pfaff et al. (ICLR 2021) — MeshGraphNets
  * Brandstetter et al. (NeurIPS 2022) — Geometric quantities for E(3) GNNs
  * Raissi et al. (Science 2019) — Physics-informed neural networks
  * Sitzmann et al. (NeurIPS 2020) — Implicit Neural Representations
  * "Universal Physics Simulation" (arxiv 2507.09733, 2025)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .e3_equivariant import E3EquivariantMP
from .attention_reasoning import AttentionReasoningGNN
from .conservation import PhysicsDiscovery
from .pinn import PhysicsResidual
# Physics residual modules are lazy-loaded in UniversalPhysicsResidual._get_residual()


# All supported physics domains
PHYSICS_DOMAINS = [
    'gravity',          # Newtonian gravity (N-body)
    'spring',           # Hooke's law (spring chain)
    'lennard_jones',    # Molecular dynamics (LJ potential)
    'fluid',            # Navier-Stokes (SPH)
    'electromagnetism',  # Maxwell's equations
    'quantum',          # Schrödinger equation
    'heat',             # Heat equation / Fourier's law
    'relativistic',     # Special relativistic mechanics
    'thermo_ideal',     # Ideal gas thermodynamics
]

# Domain grouping for multi-task learning
DOMAIN_GROUPS = {
    'particle': ['gravity', 'spring', 'lennard_jones', 'electromagnetism', 'relativistic'],
    'continuum': ['fluid', 'heat', 'quantum'],
    'thermo': ['thermo_ideal'],
}


class DomainEmbedding(nn.Module):
    """Learned embedding for each physics domain.
    
    Allows the model to condition its predictions on the domain
    while sharing parameters across domains for transfer learning.
    """
    
    def __init__(self, n_domains: int = 9, dim: int = 32):
        super().__init__()
        self.domain_emb = nn.Embedding(n_domains, dim)
        self.domain_to_idx = {d: i for i, d in enumerate(PHYSICS_DOMAINS)}
    
    def forward(self, domain: str) -> torch.Tensor:
        idx = self.domain_to_idx.get(domain, 0)
        return self.domain_emb(torch.tensor(idx, device=self.domain_emb.weight.device))


class UniversalPhysicsResidual(nn.Module):
    """Router for domain-specific PINN losses.
    
    Instantiates the appropriate physics residual module based on domain.
    """
    
    def __init__(self):
        super().__init__()
        self._cache = {}
    
    def _get_residual(self, domain: str) -> nn.Module:
        if domain in self._cache:
            return self._cache[domain]
        
        # Lazy-load physics classes on first use
        from .physics.fluids import NavierStokesResidual
        from .physics.electromagnetism import MaxwellResidual
        from .physics.quantum import SchrodingerResidual
        from .physics.thermodynamics import HeatEquationResidual
        from .physics.relativistic import RelativisticResidual
        
        if domain == 'fluid':
            res = NavierStokesResidual()
        elif domain == 'electromagnetism':
            res = MaxwellResidual()
        elif domain == 'quantum':
            res = SchrodingerResidual()
        elif domain == 'heat':
            res = HeatEquationResidual()
        elif domain == 'relativistic':
            res = RelativisticResidual()
        elif domain == 'thermo_ideal':
            res = HeatEquationResidual()
        else:
            res = PhysicsResidual(domain)
        
        self._cache[domain] = res
        return res
    
    def forward(self, domain: str, **kwargs) -> torch.Tensor:
        res = self._get_residual(domain)
        
        if domain == 'fluid':
            return res(kwargs['vel_pred'], kwargs['vel_curr'],
                       kwargs['pos'], kwargs['edge_index'], kwargs['dt'])['total']
        elif domain == 'electromagnetism':
            return res.physics_loss(kwargs['pos'], kwargs['vel_curr'],
                                    kwargs.get('charges', torch.ones_like(kwargs['pos'][:, 0])),
                                    kwargs['edge_index'], kwargs['dt'])
        elif domain == 'quantum':
            return res.tdse_residual(
                kwargs['psi_real'], kwargs['psi_imag'],
                kwargs.get('psi_real_pred', kwargs['psi_real']),
                kwargs.get('psi_imag_pred', kwargs['psi_imag']),
                kwargs['pos'].squeeze(-1), kwargs.get('V', torch.zeros_like(kwargs['pos'][:, :, 0])),
                kwargs['edge_index'], kwargs['dt']
            )['tdse_loss']
        elif domain == 'heat':
            return res.physics_loss(
                kwargs['T_pred'], kwargs['T_curr'],
                kwargs['pos'].squeeze(-1), kwargs['edge_index'],
                kwargs.get('Q', torch.zeros_like(kwargs['T_curr'])),
                kwargs['dt']
            )
        elif domain == 'relativistic':
            return res.physics_loss(
                kwargs['pos_pred'], kwargs['vel_pred'],
                kwargs['pos'], kwargs['vel_curr'],
                kwargs['masses'], kwargs.get('force', torch.zeros_like(kwargs['vel_curr'])),
                kwargs['dt']
            )['total']
        else:
            return PhysicsResidual(domain).physics_loss(
                kwargs.get('pos_pred', kwargs['pos']),
                kwargs.get('vel_pred', kwargs['vel_curr']),
                kwargs.get('pos_true', kwargs['pos']),
                kwargs.get('vel_true', kwargs['vel_curr']),
                kwargs.get('masses', torch.ones(kwargs['pos'].shape[0])),
            )


class PSNUniversal(nn.Module):
    """PSN-Universal: One graph network for all physics.
    
    Args:
        n_particles: max particles (for particle systems) or grid points
        hidden: hidden width
        n_scalar: scalar feature channels
        n_heads: attention heads
        n_domains: number of physics domains
        domain_emb_dim: domain embedding dimension
    
    Architecture:
        1. Domain embedding → conditions all modules
        2. E(3)-equivariant backbone (works for both particle and field systems)
        3. Domain-conditioned attention reasoning
        4. Universal conservation discovery (learned per domain group)
        5. Multi-domain PINN router
    """
    
    def __init__(self, n_particles: int = 100, hidden: int = 128,
                 n_scalar: int = 16, n_heads: int = 4,
                 n_domains: int = 9, domain_emb_dim: int = 32):
        super().__init__()
        self.n_particles = n_particles
        self.n_scalar = n_scalar
        self.n_heads = n_heads
        self.pos_dim = 3
        
        # Domain embedding
        self.domain_emb = DomainEmbedding(n_domains, domain_emb_dim)
        
        # E(3)-equivariant backbone (shared across all domains)
        self.equiv_encoder = E3EquivariantMP(
            n_scalar=n_scalar, hidden=hidden,
            n_layers=4, n_vec_channels=4)
        
        # Attention reasoning (domain-conditioned)
        self.attention_reasoning = AttentionReasoningGNN(
            n_scalar=n_scalar + domain_emb_dim, hidden=hidden,
            n_heads=n_heads, out_dim=self.pos_dim)
        
        # Conservation discovery (per domain group)
        self.conservation_heads = nn.ModuleDict({
            group: PhysicsDiscovery(n_scalar + domain_emb_dim, 4, hidden, 3)
            for group in DOMAIN_GROUPS
        })
        
        # Domain-specific readout heads
        self.pos_head = nn.Sequential(
            nn.Linear(n_scalar + domain_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 3),  # position/acceleration prediction
        )
        
        # Scalar field head (for T, ψ, etc.)
        self.scalar_head = nn.Sequential(
            nn.Linear(n_scalar + domain_emb_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),  # real + imag for quantum, T + dT for heat
        )
        
        # Gate: learn to balance equivariant vs attention pathways
        self.gate = nn.Sequential(
            nn.Linear((n_scalar + domain_emb_dim) * 2, hidden),
            nn.SiLU(), nn.Linear(hidden, 1),
        )
        
        # Physics router
        self.physics_router = UniversalPhysicsResidual()
    
    def _build_features(self, pos, vel, masses, domain, **kwargs):
        """Build rotation-invariant + equivariant features.
        
        Adapts feature extraction based on domain.
        """
        B, N, D = pos.shape
        dom_emb = self.domain_emb(domain)  # (emb_dim,)
        
        speeds = vel.norm(dim=-1, keepdim=True)
        ke = 0.5 * masses.unsqueeze(-1) * vel.pow(2).sum(-1, keepdim=True)
        centroid = pos.mean(dim=1, keepdim=True)
        dist_centroid = (pos - centroid).norm(dim=-1, keepdim=True)
        
        # Pairwise distance features
        pdist = torch.cdist(pos, pos)
        eye = torch.eye(N, dtype=torch.bool, device=pos.device).unsqueeze(0)
        pdist_masked = pdist.masked_fill(eye, 1e6)
        mean_dist = pdist_masked.mean(dim=-1, keepdim=True)
        min_dist = pdist_masked.min(dim=-1).values.unsqueeze(-1)
        inv_dist = (1.0 / pdist_masked.clamp(min=0.1)).mean(dim=-1, keepdim=True)
        
        # Additional domain-specific features
        extra_feats = []
        if domain == 'electromagnetism' and 'charges' in kwargs:
            charges = kwargs['charges']
            if charges.dim() == 1:
                charges = charges.unsqueeze(0).expand(B, -1)
            extra_feats.append(charges.unsqueeze(-1))
        elif domain == 'thermo_ideal':
            if 'T' in kwargs:
                extra_feats.append(kwargs['T'].unsqueeze(-1))
        
        s = torch.cat([masses.unsqueeze(-1), speeds, ke, dist_centroid,
                        mean_dist, min_dist, inv_dist] + extra_feats, dim=-1)
        
        # Pad/truncate to n_scalar
        target_dim = self.n_scalar
        if s.shape[-1] < target_dim:
            s = torch.cat([s, torch.zeros(B, N, target_dim - s.shape[-1],
                                           device=pos.device)], dim=-1)
        else:
            s = s[:, :, :target_dim]
        
        s = torch.log1p(s.clamp(min=0))
        
        # Append domain embedding to each node
        dom_emb_expanded = dom_emb.unsqueeze(0).unsqueeze(1).expand(B, N, -1)
        s = torch.cat([s, dom_emb_expanded], dim=-1)
        
        # Vector features
        pos_c = pos - centroid
        vel_c = vel - vel.mean(dim=1, keepdim=True)
        
        # Vector features: build as many channels as the encoder expects
        # E3 encoder uses 4 vector channels (each 3D = 12 dims total)
        def pad3(x):
            if x.shape[-1] == 3:
                return x
            return torch.cat([x, torch.zeros_like(x[:, :, :1])], dim=-1)
        
        n_vec_enc = 4
        vec_channels = []
        for i in range(n_vec_enc):
            v_ch = pad3(pos_c) * (1.0 if i % 2 == 0 else 0.1) + \
                   pad3(vel_c) * (0.1 if i % 2 == 0 else 1.0)
            vec_channels.append(v_ch)
        
        v = torch.cat(vec_channels, dim=-1)  # (B, N, n_vec_enc * 3)
        
        return s, v
    
    def _build_graph(self, pos, B, N, radius=None):
        """Build graph edges.
        
        For particle systems: fully connected (km from N² to N*k with radius).
        For grid/continuum: k-nearest neighbours.
        """
        dev = pos.device
        
        if radius is not None:
            # Radius-based neighbours
            pdist = torch.cdist(pos.reshape(B * N, D := pos.shape[2]),
                                pos.reshape(B * N, D))
            mask = (pdist < radius) & (pdist > 1e-8)
            src_local, dst_local = torch.where(mask)
        else:
            # Fully connected
            pairs = [(i, j) for i in range(N) for j in range(N) if i != j]
            if not pairs:
                return torch.zeros(2, 0, dtype=torch.long, device=dev), B * N
            src_local = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=dev)
            dst_local = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=dev)
        
        base = (torch.arange(B, device=dev).unsqueeze(1) * N).expand(B, len(src_local))
        src = (base + src_local.unsqueeze(0)).reshape(-1)
        dst = (base + dst_local.unsqueeze(0)).reshape(-1)
        
        return torch.stack([src, dst], dim=0), B * N
    
    def forward(self, pos, vel, masses, domain, **kwargs):
        """Universal forward pass.
        
        Args:
            pos: (B, N, 3) positions
            vel: (B, N, 3) velocities
            masses: (B, N) masses/charges/weights
            domain: str, one of PHYSICS_DOMAINS
            **kwargs: domain-specific inputs (charges, T, V, psi, etc.)
        
        Returns:
            predictions: dict with domain-specific outputs
            aux: dict with attention weights, gate values, etc.
        """
        B, N, D = pos.shape
        s, v = self._build_features(pos, vel, masses, domain, **kwargs)
        edge_index, n_nodes = self._build_graph(pos, B, N)
        
        # Split: core scalar features for E(3) encoder, domain embedding for later
        # s has shape (B, N, n_scalar + domain_emb_dim)
        n_core = self.n_scalar
        s_core = s[:, :, :n_core]  # (B, N, n_scalar) for E(3) encoder
        
        # E(3) pathway
        s_flat = s_core.reshape(B * N, n_core)
        v_flat = v.view(B * N, -1)
        x_flat = pos.reshape(B * N, 3)
        
        s_enc, _, v_enc = self.equiv_encoder(s_flat, x_flat, v_flat,
                                              edge_index, B * N)
        s_enc = s_enc.view(B, N, n_core)
        v_enc = v_enc.view(B, N, 4, 3)
        
        # Append domain embedding to encoded features for downstream heads
        dom_emb = self.domain_emb(domain).unsqueeze(0).unsqueeze(1).expand(B, N, -1)
        s_enc_full = torch.cat([s_enc, dom_emb], dim=-1)  # (B, N, n_scalar + dom_emb_dim)
        
        coeffs = self.pos_head(s_enc_full)  # (B, N, 3)
        
        # Attention pathway (also needs domain embedding)
        s_flat_full = torch.cat([s_flat, dom_emb.view(B*N, -1)], dim=-1)
        accel_attn, attn_weights, edge_weights = self.attention_reasoning(
            s_flat_full, x_flat, edge_index, B * N)
        accel_attn = accel_attn.view(B, N, D)
        
        # Scalar output (heat temp, quantum wavefunction, etc.)
        scalar_out = self.scalar_head(s_enc_full)  # (B, N, 2)
        
        # Gate
        gate = torch.sigmoid(self.gate(
            torch.cat([s, s_enc_full], dim=-1)))
        
        # Position/acceleration prediction
        accel_pred = gate * coeffs + (1 - gate) * accel_attn
        
        # Domain-specific outputs
        predictions = {'acceleration': accel_pred}
        
        if domain in ('heat', 'thermo_ideal'):
            T_pred = scalar_out[:, :, 0]
            predictions['temperature'] = T_pred
        elif domain == 'quantum':
            psi_r_pred = scalar_out[:, :, 0]
            psi_i_pred = scalar_out[:, :, 1]
            predictions['psi_real'] = psi_r_pred
            predictions['psi_imag'] = psi_i_pred
        
        aux = {
            'gate': gate,
            'attention_weights': attn_weights,
            'edge_weights': edge_weights,
            'scalar_features': s_enc,
            'vector_features': v_enc,
        }
        
        return predictions, aux
    
    def physics_loss(self, predictions, inputs, domain, **kwargs):
        """Compute domain-specific PINN loss."""
        return self.physics_router(domain, **{**predictions, **inputs, **kwargs})