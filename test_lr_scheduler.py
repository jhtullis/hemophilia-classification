"""
test_lr_scheduler.py — Plot scheduled LR values for CosineAnnealingWarmRestartsF.

Compares all five configs (v0, v1a, v1b, v1c) over 1000 epochs.
Output: test_output/lr_scheduler_comparison.png
"""

import math
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from lr_schedulers import CosineAnnealingWarmRestartsF


def get_lr_curve(T_0, T_mult, eta_min, base_lr, n_epochs=1000):
    """Simulate scheduler.step(epoch) for epochs 0..n_epochs-1."""
    dummy = nn.Linear(1, 1)
    opt = torch.optim.SGD(dummy.parameters(), lr=base_lr)
    sched = CosineAnnealingWarmRestartsF(opt, T_0=T_0, T_mult=T_mult, eta_min=eta_min)
    lrs = []
    for epoch in range(n_epochs):
        sched.step(epoch)
        lrs.append(opt.param_groups[0]["lr"])
    return lrs


def main():
    os.makedirs("test_output", exist_ok=True)

    BASE_LR = 1e-3
    N = 1000

    configs = {
        "v0  (T_mult=2,   η_min=1e-6)": dict(T_0=100, T_mult=2,   eta_min=1e-6),
        "v1a (T_mult=1.5, η_min=1e-4)": dict(T_0=100, T_mult=1.5, eta_min=1e-4),
        "v1b (T_mult=2,   η_min=1e-6)": dict(T_0=100, T_mult=2,   eta_min=1e-6),  # same schedule as v0
        "v1c (T_mult=1.5, η_min=1e-4)": dict(T_0=100, T_mult=1.5, eta_min=1e-4),  # same schedule as v1a
    }

    # Compute restart epochs for each config (for vertical reference lines)
    def restart_epochs(T_0, T_mult, N):
        restarts = []
        t = T_0
        T_i = T_0
        while t < N:
            restarts.append(t)
            T_i = T_i * T_mult
            t += T_i
        return restarts

    fig, axes = plt.subplots(len(configs), 1, figsize=(12, 10), sharex=True)
    fig.suptitle("CosineAnnealingWarmRestartsF — LR by Epoch (T_0=100, base_lr=1e-3)",
                 fontsize=13, y=0.98)

    colors = ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7"]
    epochs = list(range(N))

    for ax, (label, cfg), color in zip(axes, configs.items(), colors):
        lrs = get_lr_curve(**cfg, base_lr=BASE_LR, n_epochs=N)
        ax.plot(epochs, lrs, color=color, linewidth=1.0, label=label)
        for r in restart_epochs(cfg["T_0"], cfg["T_mult"], N):
            ax.axvline(r, color="gray", linestyle="--", linewidth=0.5, alpha=0.6)
        ax.axhline(cfg["eta_min"], color="black", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("LR", fontsize=9)
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=1e-7)

    axes[-1].set_xlabel("Epoch")
    plt.tight_layout()
    out_path = "test_output/lr_scheduler_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")

    # Print cycle boundaries for v0 and v1a/c for comparison
    print("\nCycle restart epochs (up to 1000):")
    for label, cfg in configs.items():
        rs = restart_epochs(cfg["T_0"], cfg["T_mult"], N)
        print(f"  {label}: {rs}")

    # Verify LR at epoch 0 = base_lr
    lrs_v1a = get_lr_curve(T_0=100, T_mult=1.5, eta_min=1e-4, base_lr=BASE_LR)
    assert abs(lrs_v1a[0] - BASE_LR) < 1e-9, f"Expected base_lr at epoch 0, got {lrs_v1a[0]}"
    # Verify LR at eta_min boundary is close to eta_min
    lrs_v0 = get_lr_curve(T_0=100, T_mult=2, eta_min=1e-6, base_lr=BASE_LR)
    # At epoch 99 (end of first cycle), LR should be ~eta_min
    assert lrs_v0[99] < 1e-5, f"Expected LR near eta_min at cycle end, got {lrs_v0[99]}"
    print("\nAssertions passed.")


if __name__ == "__main__":
    main()
