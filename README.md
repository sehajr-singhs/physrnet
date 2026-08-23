# PSN-1: Physics Systems Network

> **P**hysics **S**ystems **N**etwork — exact \(E(3)\) equivariant neural
> architecture with attention-based reasoning and conservation law discovery.

PSN-1 learns 3D physical dynamics by combining three complementary modules:

1. **E(3)-equivariant encoder** — rotation-invariant scalar features and
   rotation-equivariant vector features through message passing. All scalar
   coefficients are functions of invariant features only, guaranteeing exact
   equivariance.
2. **Attention reasoning** — multi-head attention discovers interaction types
   (e.g., short-range repulsion vs. long-range attraction) and predicts forces
   as attention-weighted displacements. Interpretable attention patterns reveal
   the interaction graph.
3. **Conservation law discovery** — learns to predict conserved quantities
   (energy, momentum, angular momentum) from particle states, discovering what
   is conserved without prior knowledge.

## Key results (all on CPU, 191K params)

| System | MSE | Equiv Error | Drift | Gate |
|---|---|---|---|---|
| N-body Gravity (3D) | **1.3 × 10⁻¹²** | 1.8 × 10⁻⁷ | 0.008 | 0.05 |
| Spring Chain (3D) | 4.5 × 10⁻⁷ | 6.6 × 10⁻⁷ | 0.354 | 0.24 |
| Lennard-Jones (3D) | 4.0 × 10⁻⁴ | 4.9 × 10⁻⁵ | 0.227 | 0.18 |

## Ablation (N-body gravity)

| Variant | MSE | Drift |
|---|---|---|
| Full PSN-1 | 1.5 × 10⁻¹¹ | 0.008 |
| Equivariant Only | 1.8 × 10⁻¹⁰ | **0.005** |
| Reasoning Only | **6.8 × 10⁻¹²** | 0.011 |
| No PINN | 1.3 × 10⁻¹¹ | 0.007 |
| No Conservation | 1.7 × 10⁻¹¹ | 0.009 |

## Reproduce

```bash
pip install -r requirements.txt
python experiments/run_nmi.py --epochs 30 --skip-ablation
python benchmarks/make_nmi_figures.py
```

## Repository layout

```
physrnet/
  physrnet/
    e3_equivariant.py      E(3)-equivariant message passing
    equivariant.py          E(2)-equivariant message passing (v1)
    attention_reasoning.py  multi-head attention reasoning
    conservation.py         conservation law discovery
    pinn.py                 physics constraint losses
    model.py                PhysRNet v1
    model_v2.py             PSN-1 v2 (NMI quality)
    datasets.py             2D physics trajectories
    datasets_3d.py          3D physics trajectories
    training.py             training loops (v1)
    training_v2.py          training with ablations (v2)
  experiments/
    run_nmi.py              full experiment suite
  benchmarks/
    make_nmi_figures.py     paper figures
  manuscript/
    paper.tex               Nature Machine Intelligence draft
    paper_ieee.tex          IEEE conference format
    references.bib
  index.html                project page (GitHub Pages)
  results/                  committed result JSONs
  figs/                     committed figures
```

## Architecture

The model blends two exactly \(E(3)\)-equivariant pathways:

\[
    a_i = g_i \cdot a_i^{\text{equiv}} + (1 - g_i) \cdot a_i^{\text{attn}}
\]

where \(g_i\) is a rotation-invariant gate. The gate learns that different
physical systems benefit from different computational strategies.

## Author

Sehaj Randhir Singh — independent researcher.

## Papers

- Nature Machine Intelligence: `manuscript/paper.pdf`
- IEEE conference: `manuscript/paper_ieee.pdf`
- Project page: `index.html`
