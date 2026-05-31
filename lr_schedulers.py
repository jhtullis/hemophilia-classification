"""
lr_schedulers.py — Custom learning rate schedulers.

CosineAnnealingWarmRestartsF:
    Drop-in replacement for torch.optim.lr_scheduler.CosineAnnealingWarmRestarts
    that accepts T_mult as any float >= 1.0 (e.g. 1.5), not only integers.
    All other behaviour is identical to the PyTorch built-in.

    With T_0=100, T_mult=1.5 the restart epochs and cycle lengths are:
        Cycle 0: epochs   0–99    (length 100)
        Cycle 1: epochs 100–249   (length 150)
        Cycle 2: epochs 250–474   (length 225)
        Cycle 3: epochs 475–812   (length 337)
        ...
"""

import math

from torch.optim.lr_scheduler import LRScheduler


class CosineAnnealingWarmRestartsF(LRScheduler):
    """CosineAnnealingWarmRestarts with float T_mult support.

    Args:
        optimizer:  Wrapped optimizer.
        T_0:        Length of the first cycle (positive int).
        T_mult:     Cycle-length multiplier after each restart (float >= 1.0).
        eta_min:    Minimum learning rate (default 0).
        last_epoch: The index of the last epoch (default -1).
    """

    def __init__(
        self,
        optimizer,
        T_0: int,
        T_mult: float = 1.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if not isinstance(T_0, int) or T_0 <= 0:
            raise ValueError(f"Expected positive integer T_0, got {T_0!r}")
        if T_mult < 1.0:
            raise ValueError(f"Expected T_mult >= 1.0, got {T_mult}")
        self.T_0 = T_0
        self.T_mult = float(T_mult)
        self.eta_min = eta_min
        # Set before super().__init__ so the first step() call in super finds them.
        self.T_i = float(T_0)
        self.T_cur = float(last_epoch)
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------
    def get_lr(self) -> list:
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2.0
            for base_lr in self.base_lrs
        ]

    # ------------------------------------------------------------------
    def step(self, epoch=None) -> None:
        """Advance the schedule.

        Args:
            epoch: Explicit global epoch index for cross-job continuity
                   (mirrors the usage in train_5class_hpc.py / train_patch.py).
                   If None, increments by 1.
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0.0

        if epoch is None:
            # Incremental update
            epoch = self.last_epoch + 1.0
            self.T_cur += 1.0
            if self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.T_i *= self.T_mult
        else:
            epoch = float(epoch)
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, got {epoch}")
            if epoch >= self.T_0:
                if self.T_mult == 1.0:
                    self.T_cur = epoch % self.T_0
                    self.T_i = float(self.T_0)
                else:
                    # Cycle index n satisfies:  T_0*(T_mult^n - 1)/(T_mult-1) <= epoch
                    n = math.floor(
                        math.log(
                            epoch / self.T_0 * (self.T_mult - 1.0) + 1.0,
                            self.T_mult,
                        )
                    )
                    self.T_cur = (
                        epoch
                        - self.T_0 * (self.T_mult ** n - 1.0) / (self.T_mult - 1.0)
                    )
                    self.T_i = float(self.T_0 * self.T_mult ** n)
            else:
                self.T_i = float(self.T_0)
                self.T_cur = epoch

        self.last_epoch = math.floor(epoch)
        lrs = self.get_lr()
        self._last_lr = lrs
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group["lr"] = lr
