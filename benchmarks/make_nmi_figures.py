"""Generate NMI-quality figures for PSN-1."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

RESULTS = Path('results')
FIGS = Path('figs')
FIGS.mkdir(exist_ok=True)

def load(name):
    with open(RESULTS / f'{name}.json') as f:
        return json.load(f)


def fig1_training_curves():
    """Training curves for all three systems."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    systems = ['gravity', 'spring', 'lennard_jones']
    titles = ['N-body Gravity', 'Spring Chain', 'Lennard-Jones']
    colors = ['#226999', '#1e6b3a', '#b03a2e']

    for ax, sys, title, color in zip(axes, systems, titles, colors):
        data = load(f'nmi_{sys}')
        h = data['history']
        epochs = list(range(1, len(h['train_loss']) + 1))

        ax.plot(epochs, h['val_mse'], color=color, linewidth=1.5, label='Val MSE')
        ax.plot(epochs, h['val_drift'], color=color, linewidth=1.5,
                linestyle='--', label='Energy drift')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss / Drift')
        ax.set_title(title)
        ax.set_yscale('log')
        ax.legend()
        ax.set_ylim(bottom=1e-6)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig1_training.pdf')
    plt.savefig(FIGS / 'fig1_training.png')
    plt.close()
    print("  fig1_training done")


def fig2_equivariance():
    """Equivariance error across systems and training."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Left: equivariance by system
    systems = ['gravity', 'spring', 'lennard_jones']
    titles = ['Gravity', 'Spring', 'LJ']
    equiv_errs = []
    for sys in systems:
        data = load(f'nmi_{sys}')
        equiv_errs.append(data['test_equivariance_err'])

    ax = axes[0]
    bars = ax.bar(titles, equiv_errs, color=['#226999', '#1e6b3a', '#b03a2e'], alpha=0.8)
    ax.set_ylabel('Equivariance Error')
    ax.set_title('E(3) Equivariance Verification')
    ax.set_yscale('log')
    ax.axhline(y=1e-6, color='gray', linestyle=':', label='Machine precision')
    ax.legend()

    # Right: equivariance over training (gravity)
    data = load('nmi_gravity')
    h = data['history']
    epochs = list(range(1, len(h['val_equiv_err']) + 1))
    ax = axes[1]
    ax.plot(epochs, h['val_equiv_err'], color='#226999', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Equivariance Error')
    ax.set_title('Equivariance Over Training (Gravity)')
    ax.set_yscale('log')
    ax.axhline(y=1e-6, color='gray', linestyle=':', label='Machine precision')
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGS / 'fig2_equivariance.pdf')
    plt.savefig(FIGS / 'fig2_equivariance.png')
    plt.close()
    print("  fig2_equivariance done")


def fig3_ablation():
    """Ablation study results."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    variants = ['full', 'no_pinn', 'equiv_only', 'reason_only', 'no_conservation']
    labels = ['Full\nPSN-1', 'No\nPINN', 'Equiv\nOnly', 'Reason\nOnly', 'No\nConserv.']
    colors = ['#226999', '#45B7D1', '#1e6b3a', '#FF6B6B', '#96CEB4']

    drifts = []
    equiv_errs = []
    for v in variants:
        data = load(f'ablation_{v}')
        drifts.append(data['test_drift'])
        equiv_errs.append(data['test_equivariance_err'])

    ax = axes[0]
    bars = ax.bar(labels, drifts, color=colors, alpha=0.85)
    ax.set_ylabel('Energy Drift')
    ax.set_title('Rollout Stability (Ablation)')
    for bar, val in zip(bars, drifts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)

    ax = axes[1]
    bars = ax.bar(labels, equiv_errs, color=colors, alpha=0.85)
    ax.set_ylabel('Equivariance Error')
    ax.set_title('E(3) Equivariance (Ablation)')
    ax.set_yscale('log')
    for bar, val in zip(bars, equiv_errs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.1e}', ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig3_ablation.pdf')
    plt.savefig(FIGS / 'fig3_ablation.png')
    plt.close()
    print("  fig3_ablation done")


def fig4_gate_blending():
    """How the gate blends equivariant and reasoning pathways."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    systems = ['gravity', 'spring', 'lennard_jones']
    titles = ['N-body Gravity', 'Spring Chain', 'Lennard-Jones']
    colors = ['#226999', '#1e6b3a', '#b03a2e']

    for ax, sys, title, color in zip(axes, systems, titles, colors):
        data = load(f'nmi_{sys}')
        h = data['history']
        epochs = list(range(1, len(h['gate_mean']) + 1))

        ax.plot(epochs, h['gate_mean'], color=color, linewidth=1.5)
        ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        ax.fill_between(epochs, 0, h['gate_mean'], alpha=0.15, color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gate Value (g)')
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.annotate('Equivariant\ndominant', xy=(0.02, 0.85),
                    fontsize=7, color='gray', transform=ax.transAxes)
        ax.annotate('Reasoning\ndominant', xy=(0.02, 0.1),
                    fontsize=7, color='gray', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(FIGS / 'fig4_gate.pdf')
    plt.savefig(FIGS / 'fig4_gate.png')
    plt.close()
    print("  fig4_gate done")


def fig5_comparison():
    """Summary comparison table as figure."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')

    headers = ['System', 'MSE', 'Equiv Error', 'Drift', 'Gate', 'E(3)']
    rows = []
    for sys, label in [('gravity', 'Gravity'), ('spring', 'Spring'),
                        ('lennard_jones', 'LJ')]:
        d = load(f'nmi_{sys}')
        rows.append([
            label,
            f'{d["test_mse"]:.2e}',
            f'{d["test_equivariance_err"]:.1e}',
            f'{d["test_drift"]:.4f}',
            f'{d["mean_gate"]:.3f}',
            'exact',
        ])

    table = ax.table(cellText=rows, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#e8e8e8']*6)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('PSN-1 Results Summary', fontsize=12, fontweight='bold', pad=20)

    plt.savefig(FIGS / 'fig5_summary.pdf')
    plt.savefig(FIGS / 'fig5_summary.png')
    plt.close()
    print("  fig5_summary done")


if __name__ == '__main__':
    fig1_training_curves()
    fig2_equivariance()
    fig3_ablation()
    fig4_gate_blending()
    fig5_comparison()
    print(f"\nAll figures saved to {FIGS}/")
